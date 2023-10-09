#include "et_feeder/et_feeder_node.h"

using namespace std;
using namespace Chakra;

ETFeederNode::ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node) {
  this->node_= node;
  this->id_ = node->id();
  this->name_ = node->name();
  this->runtime_ = 0;
  this->is_host_op_ = true;
  for (int i = 0; i < node->attribute_size(); i++) {
    string attr_name = node->attribute(i).name();
    if (attr_name == "is_host_op") {
      assign_attr_val(node, i, (void *)(&is_host_op_));
    } else if (attr_name == "runtime") {
      assign_attr_val(node, i, (void *)(&runtime_));
    } else if (attr_name == "num_ops") {
      assign_attr_val(node, i, (void *)(&num_ops_));
    } else if (attr_name == "tensor_size") {
      assign_attr_val(node, i, (void *)(&tensor_size_));
    } else if (attr_name == "comm_type") {
      assign_attr_val(node, i, (void *)(&comm_type_));
    } else if (attr_name == "involved_dim") {
      assign_attr_val(node, i, (void *)(&involved_dim_));
      involved_dim_size_ = node->attribute(i).bools_size();
    } else if (attr_name == "comm_priority") {
      assign_attr_val(node, i, (void *)(&comm_priority_));
    } else if (attr_name == "comm_size") {
      assign_attr_val(node, i, (void *)(&comm_size_));
    } else if (attr_name == "comm_src") {
      assign_attr_val(node, i, (void *)(&comm_src_));
    } else if (attr_name == "comm_dst") {
      assign_attr_val(node, i, (void *)(&comm_dst_));
    } else if (attr_name == "comm_tag") {
      assign_attr_val(node, i, (void *)(&comm_tag_));
    }
  }
}

shared_ptr<ChakraProtoMsg::Node> ETFeederNode::getChakraNode() {
  return node_;
}

void ETFeederNode::addChild(shared_ptr<ETFeederNode> node) {
  // Avoid adding the same child node multiple times
  // addChild is called multiple times to resolve dependencies
  if (children_set_.find(node) != children_set_.end()) {
    return;
  }
  children_vec_.emplace_back(node);
  children_set_.emplace(node);
}

vector<shared_ptr<ETFeederNode>> ETFeederNode::getChildren() {
  return children_vec_;
}

void ETFeederNode::addDepUnresolvedParentID(uint64_t node_id) {
  dep_unresolved_parent_ids_.emplace_back(node_id);
}

vector<uint64_t> ETFeederNode::getDepUnresolvedParentIDs() {
  return dep_unresolved_parent_ids_;
}

void ETFeederNode::setDepUnresolvedParentIDs(
    vector<uint64_t> const& dep_unresolved_parent_ids) {
  dep_unresolved_parent_ids_ = dep_unresolved_parent_ids;
}

void ETFeederNode::assign_attr_val(shared_ptr<ChakraProtoMsg::Node> node, int i, void *member) {
  ChakraProtoMsg::AttributeType attr_type = node->attribute(i).type();
  switch (attr_type)  {
      case ChakraProtoMsg::AttributeType::BOOL:
        *((bool *)member) = node->attribute(i).b();
        break;
      case ChakraProtoMsg::AttributeType::FLOAT:
        *((float *)member) = node->attribute(i).f();
        break;
      case ChakraProtoMsg::AttributeType::INT:
        *((int *)member) = node->attribute(i).i();
        break;
      case ChakraProtoMsg::AttributeType::STRING:
        *((string *)member) = node->attribute(i).s();
        break;
      case ChakraProtoMsg::AttributeType::BOOLS:
        for (int j = 0; j < node->attribute(i).bools_size(); j++)
          (*((vector<bool> *)member)).push_back(node->attribute(i).bools(j));
        break;
      case ChakraProtoMsg::AttributeType::FLOATS:
        for (int j = 0; j < node->attribute(i).floats_size(); j++)
          (*((vector<float> *)member)).push_back(node->attribute(i).floats(j));
        break;
      case ChakraProtoMsg::AttributeType::INTS:
        for (int j = 0; j < node->attribute(i).ints_size(); j++)
          (*((vector<int> *)member)).push_back(node->attribute(i).ints(j));
        break;
      case ChakraProtoMsg::AttributeType::STRINGS:
        for (int j = 0; j < node->attribute(i).strings_size(); j++)
          (*((vector<string> *)member)).push_back(node->attribute(i).strings(j));
        break;
      default:
        cerr << "undefined attribute type in chakra node" << endl;
        exit(EXIT_FAILURE);
  }
}

uint64_t ETFeederNode::id() {
  return id_;
}

string ETFeederNode::name() {
  return name_;
}

bool ETFeederNode::is_host_op() {
  return is_host_op_;
}

ChakraProtoMsg::NodeType ETFeederNode::type() {
  return node_->type();
}

uint64_t ETFeederNode::runtime() {
  return runtime_;
}

uint64_t ETFeederNode::num_ops() {
  return num_ops_;
}

uint32_t ETFeederNode::tensor_loc() {
  return tensor_loc_;
}

uint64_t ETFeederNode::tensor_size() {
  return tensor_size_;
}

ChakraProtoMsg::CollectiveCommType ETFeederNode::comm_type() {
  return comm_type_;
}

uint32_t ETFeederNode::involved_dim_size() {
  return involved_dim_size_;
}

bool ETFeederNode::involved_dim(int i) {
  return involved_dim_[i];
}

uint32_t ETFeederNode::comm_priority() {
  return comm_priority_;
}

uint64_t ETFeederNode::comm_size() {
  return comm_size_;
}

uint32_t ETFeederNode::comm_src() {
  return comm_src_;
}

uint32_t ETFeederNode::comm_dst() {
  return comm_dst_;
}

uint32_t ETFeederNode::comm_tag() {
  return comm_tag_;
}
