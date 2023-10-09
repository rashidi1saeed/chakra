#!/usr/bin/env python3

import json
import logging
from typing import Any, Dict
import copy
import operator
from typing import Any, Dict, List
from xmlrpc.client import Boolean

from third_party.utils.protolib import encodeMessage as encode_message
from et_def.et_def_pb2 import (
    GlobalMetadata,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    INVALID_NODE,
    COMP_NODE,
    COMM_COLL_NODE,
    BOOL,
    FLOAT,
    UINT,
    INT,
    STRING,
    BOOLS,
    FLOATS,
    UINTS,
    INTS,
    STRINGS,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    BROADCAST,
)


class PyTorch2ChakraConverter:
    def __init__(
            self,
            input_filename: str,
            output_filename: str,
            num_dims: int,
            logger: logging.Logger
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_dims = num_dims
        self.logger = logger
        self.pre_process_ops=0
        self.post_process_ops=0
        self.ops_processed_ids=[]

    @staticmethod
    def get_node_type(node: Dict[str, Any]) -> int:
        if "cat" in node.keys() and "ncclKernel" in node["name"]:
            return COMM_COLL_NODE
        elif "cat" in node.keys():
            return COMP_NODE
        elif "c10d::" in node["name"] or "nccl:" in node["name"]:
            return COMM_COLL_NODE
        elif node["op_schema"] != "" or node["outputs"]:
            return COMP_NODE
        return INVALID_NODE
    
    @staticmethod
    def is_host_op(node: Dict[str, Any]) -> Boolean:
        if "cat" in node.keys() and "ncclKernel" in node["name"]:
            return False
        elif "cat" in node.keys():
            return False
        elif "c10d::" in node["name"] or "nccl:" in node["name"]:
            return True
        elif node["op_schema"] != "" or node["outputs"]:
            return True
        assert False

    @staticmethod
    def get_attr(
            pt_node: Dict[str, Any],
            attr_name: str,
            attr_type: int
    ) -> ChakraAttr:
        attr = ChakraAttr(name=attr_name, type=attr_type)

        if attr_name in pt_node.keys():
            if attr_type == BOOL:
                attr.b = pt_node[attr_name]
            elif attr_type == FLOAT:
                attr.f = pt_node[attr_name]
            elif attr_type == UINT:
                attr.u = pt_node[attr_name]
            elif attr_type == INT:
                attr.i = pt_node[attr_name]
            elif attr_type == STRING:
                attr.s = pt_node[attr_name]
            elif attr_type == BOOLS:
                attr.bools = pt_node[attr_name]
            elif attr_type == FLOATS:
                attr.floats = pt_node[attr_name]
            elif attr_type == UINTS:
                attr.uints = pt_node[attr_name]
            elif attr_type == INTS:
                attr.ints = pt_node[attr_name]
            elif attr_type == STRINGS:
                attr.strings = pt_node[attr_name]

        return attr

    def detect_type(self, node: Dict[str, Any]) -> str:
        if "cat" in node.keys():
            return 'gpu_operator'
        elif node["op_schema"] or node["outputs"]:
            return 'operator'
        else:
            return 'label'

    def get_comm_type(self, node: Dict[str, Any]) -> int:
        if "all_reduce" in node["name"]:
            return ALL_REDUCE
        elif "all_to_all" in node["name"]:
            return ALL_TO_ALL
        elif "all_gather" in node["name"]:
            return ALL_GATHER
        elif "reduce_scatter" in node["name"]:
            return REDUCE_SCATTER
        elif "broadcast" in node["name"]:
            return BROADCAST
        else:
            node_name = node["name"]
            raise ValueError(f"{node_name} is not supported")
        return INVALID_NODE

    # https://pytorch.org/docs/stable/tensors.html
    # https://github.com/pytorch/pytorch/blob/master/c10/util/Half.h
    def get_data_type_size(self, data_type: str) -> int:
        data_type_size_dict = {
                "Tensor(float32)": 4,
                "Tensor(float)": 4,
                "Tensor(float64)": 8,
                "Tensor(double)": 8,
                "Tensor(float16)": 2,
                "Tensor(half)": 2,
                "Tensor(bfloat16)": 2,
                "Tensor(complex64)": 8,
                "Tensor(complex128)": 16,
                "Tensor(uint8)": 1,
                "Tensor(int8)": 1,
                "Tensor(int16)": 2,
                "Tensor(short)": 2,
                "Tensor(int32)": 4,
                "Tensor(int)": 4,
                "Tensor(int64)": 8,
                "Tensor(long)": 8,
                "Tensor(c10::Half)": 2,
                "Tensor(unsigned char)": 1,
                "Tensor(long int)": 4,
        }
        try:
            data_type_size = data_type_size_dict[data_type]
            return data_type_size
        except:
            raise ValueError(f"{data_type} is unsupported")

    def get_comm_size(self, node: Dict[str, Any]) -> int:
        comm_size = 1
        for input_types in node["input_types"]:
            comm_size *= self.get_data_type_size(input_types)
        for input_shape_outer in node["input_shapes"]:
            for input_shape_inner in input_shape_outer:
                comm_size = comm_size * input_shape_inner
        return comm_size
    def get_previous_step_dep_node(
            self,
            node: ChakraNode,
            max_nodes: List[int]
    ) -> int:
        ans=-1
        for mn in max_nodes:
            if node.id > mn:
                ans=mn
            else:
                break
        return ans
    def get_total_runtime(
            self,
            pytorch_et_data: Dict[str, Any]
    ) -> int:
        ans = 0
        for pt_node in pytorch_et_data["nodes"]:
            if self.detect_type(pt_node) == 'operator' and "dur" in pt_node.keys():
                ans+=pt_node["dur"]
                self.pre_process_ops+=1
            #elif self.detect_type(pt_node) != 'operator':
            #    print ("node: "+pt_node["name"]+" is not operator")
        return ans
    def node_process_sanity_checks(
            self,
            pytorch_et_data: Dict[str, Any]
    ) -> None:
        for pt_node in pytorch_et_data["nodes"]:
            if self.detect_type(pt_node) == 'operator' and "dur" in pt_node and pt_node["id"] not in self.ops_processed_ids:
                print("Operator: "+pt_node["name"]+" with id: "+str(pt_node["id"])+", and runtime: "+str(pt_node["dur"])+" is not processed acccordingly!")
    def sum_subtree_runtimes(
            self,
            node: Dict[str, Any],
            pytorch_et_data: Dict[str, Any]
    ) -> int:
        ans=0
        if self.detect_type(node) == 'operator':
            if "dur" in node:
                ans+=node["dur"]
                if node["id"] not in self.ops_processed_ids:
                    self.post_process_ops+=1
                    self.ops_processed_ids.append(node["id"])
        for pt_node in pytorch_et_data["nodes"]:
            if pt_node['parent'] == node['id']:
                ans +=  self.sum_subtree_runtimes(pt_node,pytorch_et_data)
        return ans

    def find_gpu_kernel_childs(
            self,
            id_to_relate: int,
            cpu_node: Dict[str, Any],
            pt_gpu_node_dict: Dict[int, List[Dict[str, Any]]],
            pytorch_et_data: Dict[str, Any],
            print_logs: Boolean
    ) -> None:
        for pt_node in pytorch_et_data["nodes"]:
            if pt_node['parent'] == cpu_node['id']:
                if(print_logs):
                    print("considering node: "+pt_node['name'])
                if "cat" in pt_node.keys() and (pt_node["cat"]=="kernel" or pt_node["cat"]=="gpu_memcpy"):
                    if id_to_relate in pt_gpu_node_dict.keys():
                        pt_gpu_node_dict[id_to_relate].append(pt_node)
                    else:
                        pt_gpu_node_dict[id_to_relate]=[pt_node]
                else:
                   self.find_gpu_kernel_childs(id_to_relate,pt_node,pt_gpu_node_dict,pytorch_et_data,print_logs)
        return
    def dfs(
            self,
            node: Dict[str, Any],
            pytorch_et_data: Dict[str, Any],
            pt_node_dict: Dict[int, Dict[str, Any]],
            pt_gpu_node_dict: Dict[int, List[Dict[str, Any]]],
            root_id: int,
            print_calls: bool
    ) -> int:
        if self.detect_type(node) == 'gpu_operator':
            return -1
        elif self.detect_type(node) == 'operator':
            #uncomment this if you want to add the runtimes of all children operators to the parent operator
            #node["dur"] = self.sum_subtree_runtimes(node,pytorch_et_data)
            pt_node_dict[node['id']] = node
            self.find_gpu_kernel_childs(node["id"],node,pt_gpu_node_dict,pytorch_et_data,False)
            return node['id']
        else:
            answer=-1
            for pt_node in pytorch_et_data["nodes"]:
                if (pt_node['parent'] == node['id']) and pt_node['id']!=root_id and not (pt_node['name'].startswith("## ") and pt_node['parent']==root_id):
                    if(print_calls):
                        print("child of node: "+str(node['id'])+" is: "+str(pt_node['id']))
                    answer= max(answer,self.dfs(pt_node, pytorch_et_data, pt_node_dict,pt_gpu_node_dict,root_id,print_calls))
            return answer
    def assign_ids(
            self,
            total_assigned_ids: List[int],
            assigned_ids: Dict[int,list[int]],
            id: int
    ) -> int:
        orig_id=id
        while True:
            if id in total_assigned_ids:
                id+=1
            else:
                total_assigned_ids.append(id)
                if orig_id in assigned_ids.keys():
                    assigned_ids[orig_id].append(id)
                else:
                    assigned_ids[orig_id]=[id]
                return id
    def merge_gpu_kernels_with_cpu_kernels(
            self,
            pt_node_dict: Dict[int, Dict[str, Any]],
            pt_gpu_node_dict: Dict[int, List[Dict[str, Any]]],
            pytorch_et_data: Dict[str, Any]
    ) -> Any:
        decomposed_nodes=[]
        assigned_ids={}
        total_assigned_ids=[]
        new_pt_gpu_node_dict={}
        decomposed_nodes_dep={}
        for pt_node_id,pt_node in pt_node_dict.items():
            if pt_node_id in pt_gpu_node_dict.keys():
                pt_gpu_node_dict[pt_node_id] = sorted(pt_gpu_node_dict[pt_node_id], key=lambda kv: kv["ts"])
                for gpu_node in pt_gpu_node_dict[pt_node_id]:
                    #print("node id: "+str(pt_node["id"])+", node name: "+pt_node["name"]+", node ts: "+str(pt_node["ts"])+
                     #     ", node dur: "+str(pt_node["dur"])+", gpu_node_id: "+str(gpu_node["id"])+", gpu_node_name: "+gpu_node["name"]+", gpu_node_ts: "+str(gpu_node["ts"]))
                    assert pt_node["ts"]+pt_node["dur"]>gpu_node["ts"]
                last_ts=pt_node["ts"]
                for i in range(len(pt_gpu_node_dict[pt_node_id])+1):
                    copy_node=copy.deepcopy(pt_node)
                    copy_node["id"]=self.assign_ids(total_assigned_ids,assigned_ids,pt_node_id)
                    copy_node["name"]=copy_node["name"]+"("+str(i)+")"
                    #print("i: "+str(i)+", name: "+copy_node["name"]+", node id: "+str(copy_node["id"])+", last ts: "+str(last_ts)+", copy_node_ts: "+str(copy_node["ts"])+", copy_node_dur: "+str(copy_node["dur"]))
                    if i<len(pt_gpu_node_dict[pt_node_id]):
                        #print("          i: "+str(i)+", last ts: "+str(last_ts)+", gpu_node_ts: "+str(pt_gpu_node_dict[pt_node_id][i]["ts"])+", gpu_node_dur: "+str(pt_gpu_node_dict[pt_node_id][i]["dur"]))
                        pt_gpu_node_dict[pt_node_id][i]["id"]=self.assign_ids(total_assigned_ids,assigned_ids,pt_gpu_node_dict[pt_node_id][i]["id"])
                        assert pt_gpu_node_dict[pt_node_id][i]["ts"] > copy_node["ts"]
                        copy_node["ts"]=last_ts
                        copy_node["dur"]=pt_gpu_node_dict[pt_node_id][i]["ts"]-last_ts
                        last_ts=pt_gpu_node_dict[pt_node_id][i]["ts"]
                        new_pt_gpu_node_dict.setdefault(copy_node["id"], []).append(pt_gpu_node_dict[pt_node_id][i])
                    else:
                        copy_node["dur"]=copy_node["dur"]-(last_ts-copy_node["ts"])
                        copy_node["ts"]=last_ts
                        last_ts=copy_node["ts"]+copy_node["dur"]

                    assert copy_node["ts"]>=0 and copy_node["dur"]>0
                    if i > 0:
                        assert copy_node["ts"] > decomposed_nodes[-1]["ts"]
                        decomposed_nodes_dep[copy_node["id"]]=decomposed_nodes[-1]["id"]
                    decomposed_nodes.append(copy_node)
            else:
                pt_node["id"]=self.assign_ids(total_assigned_ids,assigned_ids,pt_node_id)
                decomposed_nodes.append(pt_node)

        merged_pt_node_dict={
            decomposed_node["id"]:decomposed_node
            for decomposed_node in decomposed_nodes
        }
        #for node in decomposed_nodes:
        #    print(node["name"])
        return merged_pt_node_dict,new_pt_gpu_node_dict,assigned_ids,decomposed_nodes_dep
    def verify_pt_node_dict_integrity(
            self,
            pt_node_dict: Dict[int, Dict[str, Any]],
            pt_gpu_node_dict: Dict[int, List[Dict[str, Any]]]
    ) -> None:

        memory={}
        for pt_node_id,pt_node in pt_node_dict.items():
            assert(pt_node_id==pt_node["id"])
            if pt_node_id in pt_gpu_node_dict.keys():
                assert len(pt_gpu_node_dict[pt_node_id])==1
            if pt_node["id"] in memory:
                print("duplicated assigned ids")
                break
            memory[pt_node["id"]]=True
    def convert_to_chakra_node(
            self,
            pt_node: Dict[str, Any]
    ) -> ChakraNode:
        ck_node = ChakraNode()
        ck_node.id = pt_node["id"]
        ck_node.name = pt_node["name"]
        ck_node.type = self.get_node_type(pt_node)
        ck_node.inputs = str(pt_node["inputs"])
        ck_node.input_shapes = str(pt_node["input_shapes"])
        ck_node.input_types = str(pt_node["input_types"])
        ck_node.outputs = str(pt_node["outputs"])
        ck_node.output_shapes = str(pt_node["output_shapes"])
        ck_node.output_types = str(pt_node["output_types"])

        attr = ChakraAttr(name="runtime", type=INT)
        if "dur" in pt_node.keys():
            attr.i = pt_node["dur"]
        else:
            attr.i = 0
        ck_node.attribute.append(attr)

        attr = ChakraAttr(name="is_host_op", type=BOOL)
        if self.is_host_op(pt_node):
            attr.b=True
        else:
            attr.b=False
        ck_node.attribute.append(attr)

        attr_names = ["fw_parent", "fw_tid", "op_schema", "parent", "seq_id", "rf_id", "scope", "tid"]
        attr_types = [INT, INT, STRING, INT, INT, INT, INT, INT]
        for attr_name, attr_type in zip(attr_names, attr_types):
            attr = self.get_attr(pt_node, attr_name, attr_type)
            ck_node.attribute.append(attr)
        return ck_node

    def convert(self) -> None:
        pt_node_dict = {}
        pt_gpu_node_dict = {}
        ck_node_dict = {}
        record_param_comms_pt_node_dict = {}
        nccl_pt_node_dict = {}
        input_storage_id_node_id_dict = {}
        input_tensor_id_node_id_dict = {}
        output_storage_id_node_id_dict = {}
        output_tensor_id_node_id_dict = {}
        step_dependency=[]

        with open(self.input_filename, "r") as pytorch_et, \
                open(self.output_filename, "wb") as chakra_et:
            pytorch_et_data = json.load(pytorch_et)

            #md = GlobalMetadata(
            #  attribute=[
            #    ChakraAttr(name="schema", type=STRING, s=pytorch_et_data["schema"]),
            #    ChakraAttr(name="pid", type=UINT, u=pytorch_et_data["pid"]),
            #    ChakraAttr(name="time", type=STRING, s=pytorch_et_data["time"]),
            #    ChakraAttr(name="start_ts", type=UINT, u=pytorch_et_data["start_ts"]),
            #    ChakraAttr(name="finish_ts", type=UINT, u=pytorch_et_data["finish_ts"])
            #  ]
            #)
            #encode_message(chakra_et, md)

            pytorch_et_data["nodes"]=sorted(pytorch_et_data["nodes"], key=lambda kv: kv["ts"])

            total_runtime=self.get_total_runtime(pytorch_et_data)
            self.logger.info("total operators runtime (including child operators) in the entire PyTorch ET is: "+str(total_runtime*1000)+" ns, and # of ops are: "+str(self.pre_process_ops))
            
            root_id=-1
            for pt_node in pytorch_et_data["nodes"]:
                if  "[pytorch|profiler|execution_graph|thread]" in pt_node["name"]:
                    root_id=pt_node["id"]
                    break
            assert root_id != -1

            for pt_node in pytorch_et_data["nodes"]:
                if (pt_node["id"] == root_id) or (pt_node['name'].startswith("## ") and pt_node['parent']==root_id):
                    dep=self.dfs(pt_node, pytorch_et_data, pt_node_dict,pt_gpu_node_dict,root_id,False)
                    if pt_node['name'].startswith("## ") and pt_node['parent']==root_id:
                        #print("phase: "+pt_node['name']+" detected")
                        step_dependency.append(dep)
            #print("****")
            step_dependency.sort()


            total_runtime = 0
            for pt_node_id in pt_node_dict:
                if self.detect_type(pt_node_dict[pt_node_id]) == 'operator' and "dur" in pt_node_dict[pt_node_id].keys():
                    total_runtime+=pt_node_dict[pt_node_id]["dur"]
            self.logger.info("And total runtime (exluding child ops) is: "+str(total_runtime))

            self.logger.info("Mergibg CPU Kernels with GPU Kernels")
            pt_node_dict,pt_gpu_node_dict,assigned_ids,decomposed_nodes_dep=self.merge_gpu_kernels_with_cpu_kernels(pt_node_dict,pt_gpu_node_dict,pytorch_et_data)
            self.verify_pt_node_dict_integrity(pt_node_dict,pt_gpu_node_dict)

            self.logger.info("Identify communication nodes")
            for pt_node in pytorch_et_data["nodes"]:
                if "record_param_comms" in pt_node["name"]:
                    #print("pt_node_parent_id is: "+str(pt_node["parent"])+" the node itself is: "+str(pt_node["id"]))
                    if pt_node["parent"] in assigned_ids.keys():
                        nodes_to_assign=assigned_ids[pt_node["parent"]]
                        for parent_id in nodes_to_assign:
                            record_param_comms_pt_node_dict.update({parent_id: pt_node})
                if "nccl:" in pt_node["name"]:
                    nccl_pt_node_dict.update({pt_node["parent"]: pt_node})
            for i in range(len(step_dependency)):
                step_dependency[i]=assigned_ids[step_dependency[i]][-1]
            step_dependency.sort()

            self.logger.info("Convert PyTorch nodes to Chakra nodes")
            for pt_node_id, pt_node in pt_node_dict.items():
                #print("node to convert: "+pt_node["name"]+", dur: "+str(pt_node["dur"]))
                for i in pt_node["inputs"]:
                    if isinstance(i, list) and len(i) == 6:
                        tensor_id = i[0]
                        storage_id = i[1]
                        if storage_id > 0:
                            input_storage_id_node_id_dict.setdefault(storage_id, []).append(pt_node["id"])
                        else:
                            input_tensor_id_node_id_dict.setdefault(tensor_id, []).append(pt_node["id"])
                for o in pt_node["outputs"]:
                    if isinstance(o, list) and len(o) == 6:
                        tensor_id = o[0]
                        storage_id = o[1]
                        if storage_id > 0:
                            output_storage_id_node_id_dict.setdefault(storage_id, []).append(pt_node["id"])
                        else:
                            output_tensor_id_node_id_dict.setdefault(tensor_id, []).append(pt_node["id"])

                if pt_node_id in pt_gpu_node_dict.keys():
                    for pt_gpu_node in pt_gpu_node_dict[pt_node_id]:
                        for i in pt_gpu_node["inputs"]:
                            if isinstance(i, list) and len(i) == 6:
                                tensor_id = i[0]
                                storage_id = i[1]
                                if storage_id > 0:
                                    input_storage_id_node_id_dict.setdefault(storage_id, []).append(pt_gpu_node["id"])
                                else:
                                    input_tensor_id_node_id_dict.setdefault(tensor_id, []).append(pt_gpu_node["id"])
                        #for o in pt_gpu_node["outputs"]:
                        #    if isinstance(o, list) and len(o) == 6:
                        #        tensor_id = o[0]
                        #        storage_id = o[1]
                        #        if storage_id > 0:
                        #            continue
                        #            output_storage_id_node_id_dict.setdefault(storage_id, []).append(pt_gpu_node["id"])
                        #        else:
                        #            output_tensor_id_node_id_dict.setdefault(tensor_id, []).append(pt_gpu_node["id"])

                ck_node = self.convert_to_chakra_node(pt_node)

                if ck_node.id in pt_gpu_node_dict.keys():
                    kernel_node=self.convert_to_chakra_node(pt_gpu_node_dict[ck_node.id][0])
                    if ck_node.type == COMM_COLL_NODE:
                        if ck_node.id in record_param_comms_pt_node_dict.keys():
                            record_param_comms_pt_node = record_param_comms_pt_node_dict[ck_node.id]
                            nccl_pt_node = nccl_pt_node_dict[record_param_comms_pt_node["id"]]
                        else:
                            nccl_pt_node = nccl_pt_node_dict[ck_node.id]

                        attr = ChakraAttr(name="comm_type", type=INT)
                        attr.i = self.get_comm_type(nccl_pt_node)
                        kernel_node.attribute.append(attr)

                        attr = ChakraAttr(name="comm_size", type=INT)
                        attr.i = self.get_comm_size(nccl_pt_node)
                        kernel_node.attribute.append(attr)

                        attr = ChakraAttr(name="involved_dim", type=BOOLS)
                        for _ in range(self.num_dims):
                            attr.bools.append(True)
                        kernel_node.attribute.append(attr)
                    kernel_node.parent.append(ck_node.id)
                    ck_node_dict[kernel_node.id] = kernel_node


                ck_node_dict[ck_node.id] = ck_node

                # Adding previous phase node dependency
                dep_node_id=self.get_previous_step_dep_node(ck_node,step_dependency)
                if (dep_node_id!=-1) and (dep_node_id not in ck_node.parent):
                    ck_node.parent.append(dep_node_id)
                #Adding decomposed nodes dependency
                if pt_node_id in decomposed_nodes_dep.keys() and decomposed_nodes_dep[pt_node_id] not in ck_node.parent:
                     ck_node.parent.append(decomposed_nodes_dep[pt_node_id])

            self.logger.info("Encode data dependency with storage IDs")
            for input_storage_id, child_node_ids in input_storage_id_node_id_dict.items():
                if input_storage_id in output_storage_id_node_id_dict:
                    parent_node_ids = output_storage_id_node_id_dict[input_storage_id]
                    for child_node_id in child_node_ids:
                        for parent_node_id in parent_node_ids:
                            child_node = ck_node_dict[child_node_id]
                            if (parent_node_id not in child_node.parent)\
                            and child_node.id != parent_node_id:
                                if parent_node_id < child_node_id:
                                    child_node.parent.append(parent_node_id)

                                # remove cycles
                                parent_node = ck_node_dict[parent_node_id]
                                if (parent_node_id in child_node.parent) and\
                                   (child_node_id in parent_node.parent):
                                   if child_node_id < parent_node_id:
                                       child_node.parent.remove(parent_node_id)
                                   else:
                                       parent_node.parent.remove(child_node_id)

            self.logger.info("Encode data dependency with tensor IDs")
            for input_tensor_id, child_node_ids in input_tensor_id_node_id_dict.items():
                if input_tensor_id in output_tensor_id_node_id_dict:
                    parent_node_ids = output_tensor_id_node_id_dict[input_tensor_id]
                    for child_node_id in child_node_ids:
                        for parent_node_id in parent_node_ids:
                            child_node = ck_node_dict[child_node_id]
                            if (parent_node_id not in child_node.parent)\
                            and child_node.id != parent_node_id:
                                if parent_node_id < child_node_id:
                                    child_node.parent.append(parent_node_id)

                                # remove cycles
                                parent_node = ck_node_dict[parent_node_id]
                                if (parent_node_id in child_node.parent) and\
                                   (child_node_id in parent_node.parent):
                                   if child_node_id < parent_node_id:
                                       child_node.parent.remove(parent_node_id)
                                   else:
                                       parent_node.parent.remove(child_node_id)

            self.logger.info("Write Chakra traces")
            memory={}
            for ck_node_id in sorted(ck_node_dict.keys()):
                if ck_node_id in memory.keys():
                    print("duplicate ck id")
                    assert(False)
                memory[ck_node_id]=True
                ck_node = ck_node_dict[ck_node_id]
                encode_message(chakra_et, ck_node)

        self.logger.info("All Chakra nodes are written to the output file")
