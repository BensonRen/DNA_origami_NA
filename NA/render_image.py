# %%
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import fresnel
import IPython
import packaging.version
import math
import PIL
import io
device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")


def render(r,box,typeid):
    '''
    box is the side length (single value), r is position, typeid is the list of particle tupe
    '''
    if ('version' not in dir(fresnel) or packaging.version.parse(
            fresnel.version.version) < FRESNEL_MIN_VERSION
            or packaging.version.parse(
                fresnel.version.version) >= FRESNEL_MAX_VERSION):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )
    central_color = fresnel.color.linear([255 / 255, 0 / 255, 0 / 255])
    constituent_color = fresnel.color.linear([93 / 255, 210 / 255, 252 / 255])

    L = box #snapshot.configuration.box[0]
    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.Sphere(scene,
                                       N=len(r),
                                       radius=1.0)#0.8)
    geometry.material = fresnel.material.Material(color=[0, 0, 0],
                                                  roughness=0.8,
                                                  primitive_color_mix=1.0)
    geometry.position[:] = r[:]
    geometry.color[typeid[:] == 2] = central_color
    geometry.radius[typeid[:] == 2] = 0.5
    geometry.color[typeid[:] != 2] = constituent_color
    geometry.outline_width = 0.4
    #box = fresnel.geometry.Box(scene, [0.5*L, 0.5*L, 0, 0, 0, 0], box_radius=.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1),
                            color=(0.8, 0.8, 0.8),
                            theta=math.pi),
        fresnel.light.Light(direction=(1, 1, 1),
                            color=(1.1, 1.1, 1.1),
                            theta=math.pi / 3)
    ]
    #scene.camera = fresnel.camera.Orthographic(position=(0,0,10),#(L * 2, L, L * 2),#
    #                                           look_at=(0, 0, -10),
    #                                           up=(0, 1, 0),
    #                                           height=125)
    geometry.outline_width = 0.12
    scene.camera = fresnel.camera.Orthographic.fit(scene)
    #fresnel.camera.Orthographic(position=(56.552063, 56.594013, 0.88), look_at=(56.552063, 56.594013, -50.0), up=(0.0, 1.0, 0.0), height=120.37699890136719)
    #scene.camera = fresnel.camera.Orthographic(height=55)
    scene.background_color = (1, 1, 1)
    #out = fresnel.preview(scene)
    #image = PIL.Image.fromarray(out[:], mode='RGBA')
    #image.save('output.png')
    return scene#IPython.display.Image(tracer.sample(scene, samples=5000)._repr_png_())



# %%
#save_folder = '/expanse/lustre/projects/ddp381/pl201/nanorod/nanocube/2D/dataset_build/royal_data/jupyter_fresnel' #save destination
save_folder = os.getcwd() #save destination

def Generate_fresnel_image(file_pos,save_file_pos):
        '''
        generate image given a file path
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We read the number of particles, the system box, and the
            # particle positions into 3 separate arrays.
            N = int(np.genfromtxt(file_pos, skip_header=2, max_rows=1)[0]) #array([7600.,   nan])[0] = 7600
            box_data = np.genfromtxt(file_pos, skip_header=5, max_rows=2)
            data_unprocessed = np.genfromtxt(file_pos, skip_header=23, invalid_raise=False)
            data =data_unprocessed[:,2:6] #keep atom_type, x, y,z
            # Remove the unwanted text rows, if exists
            data = data[~np.isnan(data).all(axis=1)]
            box_xy = box_data[:,1]-box_data[:, 0]
        typeid = data[:,0] #array contains type of atoms
        r = data[:,1:]
        box = box_xy[0]
        r_min = np.amin(r, axis=0)
        r_max = np.amax(r, axis=0)
        new_min = (box-r_max+r_min)*0.5
        for i in range(2):
            r[:,i] = r[:,i] + (new_min[i]-r_min[i])
        scene = render(r,box,typeid)
        out =fresnel.pathtrace(scene, light_samples=40,samples=15,w=800,h=800)
        #out
        PIL.Image.fromarray(out[:], mode='RGBA').save(save_file_pos+'.png')
        return

# %%
for root, dirs, files in os.walk(save_folder):
    for fname in files:
        if fname[-3:]== 'png':
            save_file_name = fname[:-4]
            read_file_name = fname[-11:-4]+'0' + '.data'
            save_file_pos = os.path.join(root,save_file_name)
            simulation_file = os.path.join(simulation_file_folder,read_file_name)
            
            #print(simulation_file)
            if os.path.isfile(simulation_file):
                Generate_fresnel_image(simulation_file,save_file_pos)
            else:
                print('wtf {}'.format(fname))

# %%



