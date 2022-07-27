"""
code related to checking for counterexamples
"""

from pathlib import Path
import gzip

import numpy as np
import onnx
import onnxruntime as ort

from vnnlib import read_vnnlib_simple, get_io_nodes

def predict_with_onnxruntime(model_def, *inputs):
    'run an onnx model'
    
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)

    #names = [o.name for o in sess.get_outputs()]

    return res[0]

def read_ce_file(ce_path):
    """get file contents"""

    if ce_path.endswith('.gz'):
        with gzip.open(ce_path, 'rb') as f:
            content = f.read().decode('utf-8')
    else:
        with open(ce_path, 'r', encoding='utf-8') as f:
            content = f.read()

    content = content.replace('\n', ' ').strip()

    return content

def is_correct_counterexample(ce_path, cat, net, prop):
    """is the counterexample correct?"""

    print(f"Checking ce path: {ce_path}")

    benchmark_repo = "/home/stan/repositories/vnncomp2022_benchmarks"
    tol = 1e-4
    
    onnx_filename = f"{benchmark_repo}/benchmarks/{cat}/onnx/{net}.onnx"
    vnnlib_filename = f"{benchmark_repo}/benchmarks/{cat}/vnnlib/{prop}.vnnlib"

    if not Path(onnx_filename).is_file():
        # try unzipping
        gz_path = f"{onnx_filename}.gz"

        if not Path(gz_path).is_file():
            print(f"WARNING: onnx and gz path don't exist: {gz_path}")
        else:
            print(f"extracting from {gz_path} to {onnx_filename}")
            
            with gzip.open(gz_path, 'rb') as f:
                content = f.read()

                with open(onnx_filename, 'wb') as fout:
                    fout.write(content)

    if not Path(vnnlib_filename).is_file():
        # try unzipping
        gz_path = f"{vnnlib_filename}.gz"

        if Path(gz_path).is_file():
            print(f"extracting from {gz_path} to {vnnlib_filename}")
            
            with gzip.open(gz_path, 'rb') as f:
                content = f.read()

                with open(vnnlib_filename, 'wb') as fout:
                    fout.write(content)

    assert Path(onnx_filename).is_file()
    assert Path(vnnlib_filename).is_file(), f"vnnlib file not found: {vnnlib_filename}"

    ################################################3333

    content = read_ce_file(ce_path)

    if len(content) < 2:
        print(f"Note: no counter example provided in {ce_path}")
        return False

    #print(f"CE CONTENT:\n{content}")
    
    assert content[0] == '(' and content[-1] == ')'
    content = content[1:-1]

    x_list = []
    y_list = []

    parts = content.split(')')
    for part in parts:
        part = part.strip()
                
        if not part:
            continue
        
        assert part[0] == '('
        part = part[1:]

        name, num = part.split(' ')
        assert name[0:2] in ['X_', 'Y_']

        if name[0:2] == 'X_':
            assert int(name[2:]) == len(x_list)
            x_list.append(float(num))
        else:
            assert int(name[2:]) == len(y_list)
            y_list.append(float(num))

    onnx_model = onnx.load(onnx_filename)

    inp, _out, input_dtype = get_io_nodes(onnx_model)
    input_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)

    x_in = np.array(x_list, dtype=input_dtype)
    flatten_order = 'C'
    x_in = x_in.reshape(input_shape, order=flatten_order)
    output = predict_with_onnxruntime(onnx_model, x_in)

    flat_out = output.flatten(flatten_order)

    expected_y = np.array(y_list)
    diff = np.linalg.norm(flat_out - expected_y, ord=np.inf)

    print(f"L-inf norm difference between onnx execution and CE file output: {diff} (limit: {tol})")

    rv = diff < tol

    if rv:
        # output matched onnxruntime, also need to check that the spec file was obeyed
        rv = is_spec_violation(onnx_model, vnnlib_filename, x_list, expected_y, tol)

        if not rv:
            print("Note: counterexample in file did not violate the specification and so was invalid!")

    return rv

def is_spec_violation(onnx_model, vnnlib_filename, x_list, expected_y, tol):
    """check that the spec file was obeyed"""

    inp, out, _ = get_io_nodes(onnx_model)

    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    box_spec_list = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)

    rv = False

    for i, box_spec in enumerate(box_spec_list):
        input_box, spec_list = box_spec
        assert len(input_box) == len(x_list), f"input box len: {len(input_box)}, x_in len: {len(x_list)}"

        inside_input_box = True

        for (lb, ub), x in zip(input_box, x_list):
            if x < lb - tol or x > ub + tol:
                inside_input_box = False
                break

        if inside_input_box:
            print(f"CE input X was inside box #{i}")
            
            # check spec
            violated = False
                
            for j, (prop_mat, prop_rhs) in enumerate(spec_list):
                vec = prop_mat.dot(expected_y)
                sat = np.all(vec <= prop_rhs + tol)

                if sat:
                    print(f"prop #{j} violated:\n{vec - prop_rhs}")
                    violated = True
                    break

            if violated:
                rv = True
                break
                
    return rv

def test():
    """test code"""

    #ce_filename = "test_ce.txt"
    #cat = "cifar100_tinyimagenet_resnet"
    #net = "TinyImageNet_resnet_medium"
    #prop = "TinyImageNet_resnet_medium_prop_idx_6461_sidx_2771_eps_0.0039"

    ce_filename = "mnist-net_256x2_prop_1_0.03.counterexample.gz"
    net = "mnist-net_256x2"
    prop = "prop_1_0.03"
    cat = "mnist_fc"
    
    res = is_correct_counterexample(ce_filename, cat, net, prop)

    if res:
        print("counter example is correct")
    else:
        print("counter example is NOT correct")
        
if __name__ == "__main__":
    test()
