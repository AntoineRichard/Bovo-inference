#!/user/bin/python3

from matplotlib import pyplot as plt
import scipy.interpolate
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import cv2
import os

####################################################
# THIS IS A EXAMPLE OF HOW TO RUN OUR NEURAL NETS
####################################################

class InferFromXray:
    def __init__(self, target_height, target_width):
        self._model = None
        self._shape = [target_width, target_height]
    
    def loadJaw(self, dotpath, name):
        with open(os.path.join(dotpath, name+'.json')) as json_file:
            dvalues = json.load(json_file)
        dvalues = dvalues['dots_image']
        x = []
        y = []
        for value in dvalues:
            x.append(value['x'])
            y.append(value['y'])
        x = np.array(x)
        y = np.array(y)
        return [x,y]

    def loadTeeth(self, path, name):
        with open(os.path.join(path,name,'xray_infos.json')) as json_file:
            dct = json.load(json_file)
        x = []
        y = []
        for xtooth in dct['teeth_roi']:
            # Verify that there is a single missing teeth
            y.append(xtooth['ymin'])
            x.append((xtooth['xmin']+xtooth['xmax'])/2)
        x = np.array(x)
        y = np.array(y)
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        return [x,y]

    def parseXray(self, xray, start, end, scale, jaw, teeth, steps):
        """
        Cuts an Xray file
        """
        panos = []
        panos_raw = []
        gt = []
        width = 2*int(64*scale/2)
        jaw_interp = scipy.interpolate.interp1d(jaw[0],jaw[1], fill_value='extrapolate')
        teeth_interp = scipy.interpolate.interp1d(teeth[0],teeth[1], fill_value='extrapolate')
        for i in range(steps):
            idx = start + int(i * (end-start)/steps)
            tmp = xray[int(teeth_interp(idx)):int(jaw_interp(idx)),int(idx-width/2):int(idx+width/2)]
            print(int(teeth_interp(idx)),int(jaw_interp(idx)))
            print(tmp.shape)
            panos_raw.append(tmp.copy())
            tmp = cv2.resize(tmp, (256, 256))
            panos.append(tmp.copy())
        return np.array(panos), panos_raw
	

    def checkPath(self, path):
        """
        Checks that the folder given by the user exists
        """
        if not os.path.exists(path):
            raise ValueError('The path to the network weigths is incorrect,\
                    check and rerun. Please provide an absolute path.')

    def loadModel(self, path):
        """
        Loads the weights of the net
        """
        self.checkPath(path)
        self._model = tf.keras.models.load_model(
                    path,
                    compile=True,
                    options=None)

    def inferNN(self, to_infer):
        """
        Runs the network

        INPUTS:
        to_infer: a batch like array of shape [BxHxWxC] with B: batch, H: height, W: width, C: channel.

        OUTPUTS:
        outputs: The output of the neural network. A numpy array of shape [BxHxW]
        """
        assert(len(to_infer.shape) == 4), 'Shape of input should be 4, found '+str(len(to_infer.shape))+'.'
        assert(to_infer.shape[1] == self._shape[1]), 'Shape do not match. Height should be '+str(self._shape[1])+' when '+str(to_infer.shape[1])+'was found.'
        assert(to_infer.shape[2] == self._shape[0]), 'Shape do not match. Width should be '+str(self._shape[0])+' when '+str(to_infer.shape[2])+'was found.'
        assert(to_infer.shape[3] == 1), 'Shape do not match. Input should only have 1 channel, found '+to_infer.shape[-1]+'.'
        outputs = self._model(to_infer, training=False)
        return np.squeeze(outputs)


if __name__ == '__main__':
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('--nn_weights', type=str)
        parser.add_argument('--height', type=int, default=265)
        parser.add_argument('--width', type=int, default=256)
        parser.add_argument('--save', type=bool, default=False)
        parser.add_argument('--to_process', type=str, default='.')
        parser.add_argument('--original_ds', type=str, default='.')
        parser.add_argument('--jaw_ds', type=str, default='.')
        parser.add_argument('--output', type=str, default='test_run-save')
        return parser.parse_args()

    args = parse()
    # Build data and model loader
    IFX = InferFromXray(256,256)
    IFX.loadModel(args.nn_weights)
    # List files to process
    scans = os.listdir(args.to_process)
    # Creates save directory
    os.makedirs(args.output,exist_ok=True)
    # Process all files in directory
    for scan in scans:
        jaw = IFX.loadJaw(args.jaw_ds, scan) # Load information regarding the bottom of the jaw
        teeth = IFX.loadTeeth(args.original_ds, scan) # Load information regarding the top of the teeth
        xray = np.load(os.path.join(args.to_process,scan,scan+'_xray.npz'))['image'] # load xray
        # Load information regarding the extrmities of the jaw in the virtual 
        with open(os.path.join(args.to_process,scan,'infos.json')) as jfile:
            infos = json.load(jfile) 
        x = []
        for i in infos:
            x.append(i['point']['x'])
        x = np.array(x)
        steps = int(np.max(x) - np.min(x))
        with open(os.path.join(args.to_process,scan,'xray_infos.json')) as jfile:
            x_infos = json.load(jfile)

        xray_x = []
        for i in x_infos:
            xray_x.append(i['point']['x'])
        xray_x = np.array(xray_x)
        # Get scale
        start = int(np.min(xray_x))
        end = int(np.max(xray_x))
        scale = (end - start)/(steps*1.0)
        # Generate crops in the panoramic image
        panos, raw = IFX.parseXray(xray, start, end, scale, jaw, teeth, steps)
        preds = []
        # RUN the net 
        for i in range(int(panos.shape[0]/64)+1):
            p = np.expand_dims(panos[i*64:(i+1)*64],axis=-1) - 0.5
            pred = IFX.inferNN(p)
            # Clip and remap
            pred = (np.clip(np.array(pred) + 0.5, 0, 1) * (2**16-1)).astype(np.uint16)
            preds.append(pred)
        preds = np.concatenate(preds,axis=0)
        # Save
        os.makedirs(os.path.join(args.output,scan),exist_ok=True)
        truth = np.load(os.path.join(args.to_process,scan,scan+'_sagittales.npz'))['arr_0']
        truth = truth[int(np.min(x)):int(np.max(x))]
        for i in range(preds.shape[0]):
           cv2.imwrite(os.path.join(args.output,scan,'pred_'+str(i)+'.png'), preds[i])
           cv2.imwrite(os.path.join(args.output,scan,'input_'+str(i)+'.png'), (raw[i]*255).astype(np.uint8))
           cv2.imwrite(os.path.join(args.output,scan,'truth_'+str(i)+'.png'), (256*truth[i]/4096).astype(np.uint8))
