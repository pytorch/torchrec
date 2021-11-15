from __future__ import absolute_import, division, print_function, unicode_literals

import ast
import csv
import os
import sys


class ops:
    is_fc_type = False
    is_conv_type = False
    order = ""
    # Convolutions
    activations_dict = {}
    filter_dict = {}
    bias_dict = {}
    outputs_dict = {}
    # FCs
    X_dict = {}
    W_dict = {}

    def compute_mnk(self):
        if self.is_conv_type is True:
            M = (
                self.activations_dict["N"]
                * self.outputs_dict["OH"]
                * self.outputs_dict["OW"]
            )
            K = (
                self.activations_dict["C"]
                * self.filter_dict["KH"]
                * self.filter_dict["KW"]
            )
            N = self.outputs_dict["M"]
            self.MNK = (M, N, K)
        elif self.is_fc_type is True:  # FC is_fc_type == True
            M = self.X_dict["M"]
            K = self.X_dict["K"]
            N = self.W_dict["N"]
            self.MNK = (M, N, K)
        return 0

    def __init__(self, row):
        self.row = row
        if row["optype"] in {"FC", "FbFCPacked", "FbFCPackedAcc16", "FCTransposed"}:
            self.is_fc_type = True
        # TO DO: ConvTranspose
        # ignoring Int8 Operators for now
        if row["optype"] in {"Conv"}:
            self.is_conv_type = True
        return


def get_unique_entries(list_of_ops):
    ret_list = []
    for entry in list_of_ops:
        ret_list.append(entry.MNK)
    return list(set(ret_list))


def print_to_csv(list_of_mnk):
    if len(list_of_mnk):
        print("M, N, K")
        for entry in list_of_mnk:
            print("{%s,%s,%s}".format(str(entry[0]), str(entry[1]), str(entry[2])))


def print_vector_of_vectors(list_of_mnk):
    if len(list_of_mnk):
        print("vector<vector<int> shapes = ")
        print("{")
        for entry in list_of_mnk:
            print("{" + "{}, {}, {}".format(entry[0], entry[1], entry[2]) + "},")
        print("}")
    return


# TO DO: parse malformed string arg_string
# to get Strides and Pad values
def get_arguments(row, new_op):
    arg_string = row["args"]
    if "NCHW" in arg_string:
        new_op.order = "NCHW"
    elif "NHWC" in arg_string:
        new_op.order = "NHWC"
    return


def get_outputs(row, new_op):
    output_string = row["outputs"]
    new_dict = ast.literal_eval(output_string)
    conv_dims = len(new_dict["0"])

    if new_op.is_conv_type is True:
        new_op.outputs_dict["N"] = new_dict["0"][0]
        new_op.outputs_dict["M"] = new_dict["0"][1]
        new_op.outputs_dict["OH"] = new_dict["0"][2]
        if conv_dims > 3:
            new_op.outputs_dict["OW"] = new_dict["0"][3]
        if conv_dims > 4:
            new_op.outputs_dict["OT"] = new_dict["0"][4]
    return 0


# We should now handle Convs using both NCHW and NHWC format
# We should also be able to handle 1,2,3-D Convolutions
# Not Handling: Int8Conv, ConvRelu
def get_inputs(new_op, row):
    input_string = row["inputs"]
    new_dict = ast.literal_eval(input_string)
    if bool(new_dict) is False or len(new_dict) != 3:
        return 0

    if new_op.is_fc_type is True:
        if len(new_dict["0"]) > 2:
            return 0
        new_op.X_dict["M"] = new_dict["0"][0]
        new_op.X_dict["K"] = new_dict["0"][1]
        if row["optype"] != "FCTransposed":
            new_op.W_dict["N"] = new_dict["1"][0]
            new_op.W_dict["K"] = new_dict["1"][1]
        else:
            new_op.W_dict["K"] = new_dict["1"][0]
            new_op.W_dict["N"] = new_dict["1"][1]

    elif new_op.is_conv_type is True:
        conv_dims = len(new_dict["0"])
        new_op.activations_dict["N"] = new_dict["0"][0]
        new_op.filter_dict["M"] = new_dict["1"][0]

        if new_op.order == "NCHW":
            new_op.activations_dict["C"] = new_dict["0"][1]
            new_op.activations_dict["H"] = new_dict["0"][2]
            new_op.filter_dict["C"] = new_dict["1"][1]
            new_op.filter_dict["KH"] = new_dict["1"][2]
            if conv_dims > 3:
                new_op.activations_dict["W"] = new_dict["0"][3]
                new_op.filter_dict["KW"] = new_dict["1"][3]
            if conv_dims > 4:
                new_op.activations_dict["T"] = new_dict["0"][4]
                new_op.filter_dict["KT"] = new_dict["1"][4]

        elif new_op.order == "NHWC":
            new_op.activations_dict["H"] = new_dict["0"][1]
            new_op.filter_dict["KH"] = new_dict["1"][1]
            if conv_dims > 3:
                new_op.activations_dict["W"] = new_dict["0"][2]
                new_op.filter_dict["KW"] = new_dict["1"][2]
            if conv_dims > 4:
                new_op.activations_dict["T"] = new_dict["0"][3]
                new_op.filter_dict["KT"] = new_dict["1"][3]

            new_op.activations_dict["C"] = new_dict["0"][conv_dims - 1]
            new_op.filter_dict["C"] = new_dict["1"][conv_dims - 1]

            return 1


# Run this script as:
# python get_matrix_shapes_from_scuba.py query.txt log.csv
# where query.txt calls scuba and log.csv is the output
#!!TO DO: Use Scuba Python Client and not rely on intermediate CSV!!
if __name__ == "__main__":
    print_csv = False
    run_scuba = True
    csv_file_name = "csv.log"
    if run_scuba is True:
        num_args = len(sys.argv)
        assert num_args == 3, "Number of Input Args to script incorrect"
        query_file_name = str(sys.argv[1])
        csv_file_name = str(sys.argv[2])
        cmd_to_run = "source " + query_file_name + " &> " + csv_file_name
        print(cmd_to_run)
        sh_file = open("run_query.sh", "w")
        sh_file.write("#!/bin/bash\n")
        sh_file.write(cmd_to_run)
        sh_file.close()
        os.system("chmod +x run_query.sh")
        os.system("./run_query.sh")

    input_file = csv.DictReader(open(csv_file_name))
    list_of_ops = []
    for row in input_file:
        new_op = ops(row)
        get_arguments(row, new_op)
        ret = get_inputs(new_op, row)
        if ret == 0:
            continue
        get_outputs(row, new_op)
        new_op.compute_mnk()
        list_of_ops.append(new_op)

    list_of_mnk = get_unique_entries(list_of_ops)
    if print_csv:
        print_to_csv(list_of_mnk)
    else:
        print_vector_of_vectors(list_of_mnk)
