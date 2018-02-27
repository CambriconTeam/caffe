/**
* @author handong
*/

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <vector>

#include "caffe/layers/proposal_layer.hpp"

#define SIZE 1000

using namespace std;

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::Sort(vector<Dtype> &scores, vector<int> &id, int start, int length) {
  if(length<2) return;
  Dtype swap;
  int swap_id;
  if(length>2) {
    int i=0;
    int j=length-1;
    while(j>i) {
      for(;j>i;j--)
        if(scores[i+start]<scores[j+start]) {
          swap = scores[i+start];
          swap_id = id[i+start];
          scores[i+start] = scores[j+start];
          id[i+start] = id[j+start];
          scores[j+start] = swap;
          id[j+start] = swap_id;
          break;
        }
      for(;i<j;i++)
        if(scores[i+start]<scores[j+start]) {
          swap = scores[i+start];
          swap_id = id[i+start];
          scores[i+start] = scores[j+start];
          id[i+start] = id[j+start];
          scores[j+start] = swap;
          id[j+start] = swap_id;
          break;
        }
    }
    Sort(scores,id,start,i+1);
    Sort(scores,id,start+i+1,length-i-1);
    return;
  } else {
    if(scores[0+start]<scores[1+start]) {
      swap = scores[0+start];
      swap_id = id[0+start];
      scores[0+start] = scores[1+start];
      id[0+start] = id[1+start];
      scores[1+start] = swap;
      id[1+start] = swap_id;
    }
    return;
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::CreateAnchor(vector<Dtype> * anchor, Dtype * init_anchor, int A, int W, int H, Dtype stride) {
  anchor -> resize(A*W*H*4);
  for(int i=0;i<H;i++)
    for(int j=0;j<W;j++)
      for(int k=0;k<A;k++) {
        (*anchor)[i*W*A*4+j*A*4+k*4+0] = j*stride + init_anchor[k*4+0];
        (*anchor)[i*W*A*4+j*A*4+k*4+1] = i*stride + init_anchor[k*4+1];
        (*anchor)[i*W*A*4+j*A*4+k*4+2] = j*stride + init_anchor[k*4+2];
        (*anchor)[i*W*A*4+j*A*4+k*4+3] = i*stride + init_anchor[k*4+3];
      }
}

template <typename Dtype>
void ProposalLayer<Dtype>::CreateBox(vector<Dtype> * box, vector<Dtype> anchor, const Dtype * delt, int A, int W, int H, Dtype im_w, Dtype im_h) {
  box -> resize(A*W*H*4);
  for(int i=0;i<H;i++)
    for(int j=0;j<W;j++)
      for(int k=0;k<A;k++) {
        int anchor_loc = i*W*A*4+j*A*4+k*4;
        Dtype x0 = anchor[anchor_loc+0];
        Dtype y0 = anchor[anchor_loc+1];
        Dtype x1 = anchor[anchor_loc+2];
        Dtype y1 = anchor[anchor_loc+3];
        int delt_loc = i*W+j+k*H*W*4;
        Dtype dx = delt[delt_loc+0*H*W];
        Dtype dy = delt[delt_loc+1*H*W];
        Dtype dw = delt[delt_loc+2*H*W];
        Dtype dh = delt[delt_loc+3*H*W];
        Dtype cx = (x0+x1+1)/2;
        Dtype cy = (y0+y1+1)/2;
        Dtype w = x1-x0+1;
        Dtype h = y1-y0+1;
        Dtype ncx = cx+dx*w;
        Dtype ncy = cy+dy*h;
        Dtype nw = exp(dw)*w;
        Dtype nh = exp(dh)*h;
        //--==--==--==--==--==--==--==-- here choice the order of box as anchor or as scores
        int box_loc = i*W*4+j*4+k*H*W*4;
        //int box_loc = anchor_loc;
        (*box)[box_loc+0] = max(ncx-nw/2,(Dtype)0);
        (*box)[box_loc+1] = max(ncy-nh/2,(Dtype)0);
        (*box)[box_loc+2] = min(ncx+nw/2,im_w-1);
        (*box)[box_loc+3] = min(ncy+nh/2,im_h-1);
        //if(anchor_loc<=18*4) {
        //  cout << x0 << " " << y0 << " " << x1 << " " << y1 << " " << dx << " " << dy << " " << dw << " "  << dh << " ";
        //  cout << cx << " " << cy << " " << w  << " " << h  << " " << ncx << " " << ncy << " " << nw << " " << nh << endl;
        //}
      } 
}

template <typename Dtype>
void ProposalLayer<Dtype>::RemoveSmallBox(vector<Dtype> box, vector<int> id, vector<int> * keep, int * keep_num, int size, Dtype w_min_size, Dtype h_min_size) {
  keep -> resize(size);
  int j = 0;
  for(int i=0;i<size;i++) {
    if((box[i*4+2]-box[i*4+0]+1)>=w_min_size && (box[i*4+3]-box[i*4+1]+1)>=h_min_size) {
      (*keep)[j] = id[i];
      j++;
    }
  }
  *keep_num = j;
}

template <typename Dtype>
void ProposalLayer<Dtype>::GetNewScoresByKeep(const Dtype * scores, vector<Dtype> * new_scores,vector<int> keep, int keep_num) {
  new_scores -> resize(keep_num);
  for(int i=0;i<keep_num;i++)
    (*new_scores)[i] = scores[keep[i]];
}

template <typename Dtype>
void ProposalLayer<Dtype>::GetTopScores(vector<Dtype> &scores, vector<int> &id, int * size, int THRESH) {
  Sort(scores,id,0,*size);
  *size = min(*size,THRESH);
}

template <typename Dtype>
void ProposalLayer<Dtype>::NMS(vector<Dtype> box, vector<int> &id, int * id_size, Dtype THRESH, int MAX_NUM) {
  vector<bool> is_used (*id_size,true);
  vector<Dtype> area (*id_size);
  for(int i=0;i<*id_size;i++) {
    area[i] = (box[id[i]*4+2]-box[id[i]*4+0]+1)*(box[id[i]*4+3]-box[id[i]*4+1]+1);
  }
  int j=0;
  for(int i=0;i<*id_size;i++) {
    if(is_used[i]) {
      id[j] = id[i];
      for(int k=i+1;k<*id_size;k++) {
        if(is_used[k]) {
          Dtype inter_x1 = max(box[id[i]*4+0],box[id[k]*4+0]);
          Dtype inter_y1 = max(box[id[i]*4+1],box[id[k]*4+1]);
          Dtype inter_x2 = min(box[id[i]*4+2],box[id[k]*4+2]);
          Dtype inter_y2 = min(box[id[i]*4+3],box[id[k]*4+3]);
          Dtype inter_area = max((Dtype)0,inter_x2-inter_x1+1)*max((Dtype)0,inter_y2-inter_y1+1);
          Dtype over = inter_area/(area[i]+area[k]-inter_area);
          if(over>THRESH)
            is_used[k] = false;
        }
      }
      j=j+1;
      if(j>=MAX_NUM) break;
    }
  }
  *id_size = j;
}

template <typename Dtype>
void ProposalLayer<Dtype>::GetNewBox(vector<Dtype> box, Dtype * new_box, vector<int> id, int id_size) {
  //*new_box = (Dtype *)malloc(id_size*sizeof(Dtype));
  for(int i=0;i<id_size;i++) {
    new_box[i*5+0]=0;
    new_box[i*5+1]=box[id[i]*4+0];
    new_box[i*5+2]=box[id[i]*4+1];
    new_box[i*5+3]=box[id[i]*4+2];
    new_box[i*5+4]=box[id[i]*4+3];
  }
}

template <typename Dtype>
int ProposalLayer<Dtype>::Proposal(const Dtype * bbox_pred, const Dtype * scores, int H, int W, int A, Dtype * init_anchor, Dtype stride,Dtype im_w,Dtype im_h,Dtype im_min_w,Dtype im_min_h, Dtype top_thresh, Dtype nms_thresh, int nms_num, Dtype * new_box) {
  vector<Dtype> anchor;
  //cout << "create anchor" << endl;
  CreateAnchor(&anchor,init_anchor,A,W,H,stride);
  vector<Dtype> box;
  //cout << "create box" << endl;
  CreateBox(&box,anchor,bbox_pred,A,W,H,im_w,im_h);
  int total = A*W*H;
  vector<int> id (total);
  for(int i=0;i<total;i++)
    id[i]=i;
  vector<int> keep;
  int keep_num=0;
  //cout << "remove small box" << endl;
  RemoveSmallBox(box,id,&keep,&keep_num,total,im_min_w,im_min_h);
  //cout << keep[0] << endl;
  vector<Dtype> new_scores;
  //cout << "get new scores" << endl;
  GetNewScoresByKeep(scores,&new_scores,keep,keep_num);
  //cout << "get top scores" << endl;
  GetTopScores(new_scores,keep,&keep_num,top_thresh);
  //cout << keep[0] << endl;
  //cout << "nms" << endl;
  NMS(box,keep,&keep_num,nms_thresh,nms_num);
  //cout << "get new box" << endl;
  GetNewBox(box,new_box,keep,keep_num);
  //cout << "finish" << endl;
  //for (int i=0;i<18*5;i++) {
  //  cout << new_box[i] << " ";
  //  if(i%5==4)
  //    cout << endl;
  //}
  return keep_num;
}

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ProposalParameter proposal_param = this->layer_param().proposal_param();
  stride_ = proposal_param.stride();
  im_min_w_ = proposal_param.im_min_w();
  im_min_h_ = proposal_param.im_min_h();
  top_num_ = proposal_param.top_num();
  nms_thresh_ = proposal_param.nms_thresh();
  nms_num_ = proposal_param.nms_num();
  A_ = proposal_param.anchor_num();
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(nms_num_,5,1,1);
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype * scores = bottom[0]->cpu_data() + bottom[0]->offset(0,A_);
  const Dtype * bbox_pred = bottom[1]->cpu_data();
  const Dtype * im_info = bottom[2]->cpu_data();
  Dtype * top_data = top[0]->mutable_cpu_data();
  Dtype init_anchor[] = {-84,-40,99,55,-176,-88,191,103,-360,-184,375,199,-56,-56,71,71,-120,-120,135,135,-248,-248,263,263,-36,-80,51,95,-80,-168,95,183,-168,-344,183,359};
  Dtype im_h = im_info[0];
  Dtype im_w = im_info[1];
  Dtype scale = im_info[2];
  Dtype H = bottom[0]->height();
  Dtype W = bottom[0]->width();
  int box_num = Proposal(bbox_pred,scores,H,W,A_,init_anchor,stride_,im_w,im_h,im_min_w_*scale,im_min_h_*scale,top_num_,nms_thresh_,nms_num_,top_data);
  for(int i=box_num*5;i < nms_num_*5;i++)
    top_data[i] = 0;
}

//template <typename Dtype>
//void ProposalLayer<Dtype>::Forward_gpu(
//    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  Forward_cpu(bottom,top);
//}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

//template <typename Dtype>
//void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//}

//#ifdef CPU_ONLY
//STUB_GPU(ProposalLayer);
//#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
