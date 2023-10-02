import argparse
import numpy as np
import sys
import random
import time

import tritonclient.grpc as tritongrpcclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=False,
                        default="model1",
                        #default="model2",
                        #default="modelEnsemble",
                        help='Model name')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-r',
                        '--repeats',
                        type=int,
                        required=False,
                        dest='repeats',
                        default=20,
                        help='Number of repeats')
    
    FLAGS = parser.parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url,
                                                               verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = FLAGS.model_name 
    
    mconf = triton_client.get_model_config(model_name, as_json=True)
    print('config:\n', mconf)
    inputs = {}
    dtypes = {}
    
    npf = 50
    nsv = 10
    
    np.random.seed(1234)
    
    for inp in mconf['config']['input']:
        name = inp['name']
        shape = [int(i) for i in inp['dims']]
        dtype = inp['data_type']
        dtypes[name] = dtype[5:]
        
        # first dimension is the batch size
        #shape = [1] + shape
        shape[0] = 2
        
        inputs[name] = np.random.random(shape).astype(np.float32)
        
    print("\n\nINPUT\n\n")
    print(inputs)
    
    tritoninputs = []
    for name in inputs.keys():
        tritoninputs.append(tritongrpcclient.InferInput(name, inputs[name].shape, dtypes[name]))
        # prepare inputs
        tritoninputs[-1].set_data_from_numpy(inputs[name])
        
    # prepare outputs
    for i in range(FLAGS.repeats):
        outputs = []
        
        if FLAGS.model_name != "modelEnsemble":
            outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT__0"))
        else:
            outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT__model1__0"))
            outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT__model2__0"))
        
        # get the output
        results = triton_client.infer(model_name=model_name,
                                      inputs=tritoninputs,
                                      outputs=outputs)
        print("\n\nOUTPUT:")
        if FLAGS.model_name != "modelEnsemble":
            print(results.as_numpy("OUTPUT__0"))
        else:
            print(results.as_numpy("OUTPUT__model1__0"))
            print(results.as_numpy("OUTPUT__model2__0"))
        
        time.sleep(0.3)
