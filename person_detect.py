
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        print("Model initialized")

    def load_model(self, device="CPU", cpu_extension=None):
        '''
        TODO: This method needs to be completed by you
        '''

        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        #model_xml = model
        #model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.core = IECore()

        # Add a CPU extension, if applicable
        #if cpu_extension and "CPU" in device:
        #    self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.exec_net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print('Network loaded...')



        # Get the input layer
        #self.input_blob = next(iter(self.exec_net.inputs))
        #self.output_blob = next(iter(self.exec_net.outputs))
        return
   

    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        #p_frame = preprocessing(image, height, width)
        p_frame = self.preprocess_input(image)



        #def exec_net(self, net_inputs):
        ### TODO: Start an asynchronous request ###
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        input_dict = {self.input_name: p_frame}
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs=input_dict)
        
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            result = infer_request_handle.outputs[self.output_name]
        
        #result = result['detection_out']
        #result = self.preprocess_outputs(result)
        #self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        #def get_output(self):
        ### TODO: Extract and return the output results
        '''
        Returns a list of the results for the output layer of the network.
        '''
        coordinates, out_frame = self.draw_outputs(result, p_frame)
        return coordinates, image
    

        ############
        #def predict(self, image, w, h):
        #input_img = image
        #image=self.preprocess_input(image)
        #input_dict = {self.input_name:image}
        #infer_request_handle = self.ex_net.start_async(request_id=0, inputs=input_dict)
        #infer_status = infer_request_handle.wait()
        #if infer_status == 0:
            #res = infer_request_handle.outputs[self.output_name]
            
        #coords, image = self.draw_outputs(res, input_img, w, h)    
        #return coords, input_img
    
    
    
    
    
    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        '''
        width = int(image.shape[1]) 
        height = int(image.shape[0])
        detections = []
        
        for box in coords[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        return coords, image                


        
    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        '''
        output = np.squeeze(outputs)
        return output

    
    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        try:
            n, c, height, width = self.input_shape
            #input_img = np.copy(image)
            input_img= cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            input_img = input_img.transpose(2,0,1)
            input_img = input_img.reshape((n, c, self.input_shape[2], self.input_shape[3]))
        except Exception as e:
            print('Preprocess inputs error: ',e)
        return input_img
    


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)