#ifndef CAFFE_PROPOSAL_LAYER_HPP_
#define CAFFE_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace std;
namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
 public:
  explicit ProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Proposal"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype stride_,im_min_w_,im_min_h_,top_num_,nms_thresh_,nms_num_;
  int A_;

  void Sort(vector<Dtype> &scores, vector<int> &id,int start, int length);
  void CreateAnchor(vector<Dtype> * anchor, Dtype * init_anchor, int A, int W, int H, Dtype stride);
  void CreateBox(vector<Dtype> * box, vector<Dtype >anchor, const Dtype * delt, int A, int W, int H, Dtype im_w, Dtype im_h);
  void RemoveSmallBox(vector<Dtype> box, vector<int> id,vector< int >* keep, int * keep_num, int size, Dtype w_min_size, Dtype h_min_size);
  void GetNewScoresByKeep(const Dtype * scores, vector<Dtype> * new_scores,vector<int >keep, int keep_num);
  void GetTopScores(vector<Dtype> &scores,vector< int > &id, int * size, int THRESH);
  void NMS(vector<Dtype> box,vector< int > &id, int * id_size, Dtype THRESH, int MAX_NUM);
  void GetNewBox(vector<Dtype> box, Dtype * new_box, vector<int> id, int id_size);
  int Proposal(const Dtype * bbox_pred, const Dtype * scores, int H, int W, int A, Dtype * init_anchor, Dtype stride,Dtype im_w,Dtype im_h,Dtype im_min_w,Dtype im_min_h, Dtype top_thresh, Dtype nms_thresh, int nms_num, Dtype * new_box);
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_

