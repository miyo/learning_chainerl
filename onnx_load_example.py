# -*- coding: utf-8 -*-
import onnx
import sys

def main(path):
    # ONNX形式のモデルを読み込む
    model = onnx.load(path)

    print("ir_version=", model.ir_version)
    print("prducer_name=", model.producer_name, ", producer_version=", model.producer_version)
    print("domain=", model.domain)
    print("model_version=", model.model_version)
    

    # モデル（グラフ）を構成するノードを全て出力する
    print("====== Nodes ======")
    for i, node in enumerate(model.graph.node):
        print("[Node #{}]".format(i))
        print(node)

    # モデルの入力データ一覧を出力する
    print("====== Inputs ======")
    for i, input in enumerate(model.graph.input):
        print("[Input #{}]".format(i))
        print(input)

    # モデルの出力データ一覧を出力する
    print("====== Outputs ======")
    for i, output in enumerate(model.graph.output):
        print("[Output #{}]".format(i))
        print(output)

    print("====== Initializers ======")
    for i, ini in enumerate(model.graph.initializer):
        print("---")
        print("[Initializer #{}]".format(i))
        print(ini)
        print("type name=", type(ini).__name__)
        print("data type=", ini.data_type)
        print("dims=", ini.dims)
        print("doc string=", ini.doc_string)
        print("double_data=", ini.double_data)
        print("float_data=", ini.float_data)
        print("int32_data=", ini.int32_data)
        print("int64_data=", ini.int64_data)
        print("name=", ini.name)
        print("raw_data=", ini.raw_data)
        print("segment=", ini.segment)
        print("string_data=", ini.string_data)
        print("uint64_data=", ini.uint64_data)
        print("---")

if __name__ == "__main__":
    main(sys.argv[1])
