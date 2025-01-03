import gradio as gr
import subprocess
import sys
import os
import pathlib
import shutil
from time import sleep
from xml.etree.ElementTree import parse
from tqdm import tqdm


PointConfig=""
DepthConfig=""
LightConfig=""
TrimeshConfig=""
WrpConfig=""
ply=""
rgb=""
depth=""
light=""
trimesh=""
wrpply=""

def plyfile(file, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Uploading Starting")
    file_paths = file.name

    for i in progress.tqdm(file_paths, desc="Uploading from file"):
        #sleep(0.1)
        progress(1)

    global ply
    ply = file_paths

    return file_paths

def rgbfile(file, progress=gr.Progress(track_tqdm="True")):
    progress(0, desc="Uploading Starting") 
    rgbfile_paths = file.name
    
    for i in progress.tqdm(rgbfile_paths, desc="Uploading from file"):
        #sleep(0.1)
        progress(1)

    rgbfile_name = os.path.basename(rgbfile_paths).split('/')[-1]    
    rgb_name = os.path.splitext(rgbfile_name)

    global rgb
    rgb = rgb_name[0]
    currentPath = os.getcwd()

    if os.path.exists(rgbfile_paths):
        shutil.move(rgbfile_paths , currentPath+"//DepthMap//"+rgbfile_name)
    yield rgbfile_paths


def depthfile(file, progress=gr.Progress(track_tqdm="True")):
    depthfile_paths = file.name
    progress(0, desc="Uploading Starting")
    
    for i in progress.tqdm(depthfile_paths, desc="Uploading from file"):
        #sleep(0.01)
        progress(1)

    depthfile_name = os.path.basename(depthfile_paths).split('/')[-1]
    depth_name = os.path.splitext(depthfile_name)
    
    global depth
    depth = depth_name[0]

    currentPath = os.getcwd()

    if os.path.exists(depthfile_paths):
        shutil.move(depthfile_paths , currentPath+"//DepthMap//"+depthfile_name)
    return depthfile_paths

def lightfile(directory, progress=gr.Progress(track_tqdm="True")):
    progress(0, desc="lightfile Uploading Starting")
    
    for file in directory:
        lightfile_name = os.path.basename(file.name).split('/')[-1]
        light_name = os.path.splitext(lightfile_name)
        currentPath = os.getcwd()
        if os.path.exists(file.name):
            shutil.move(file.name , currentPath+"//LightField//"+light_name[0])
   
    global light
    light = os.getcwd()+"//LightField//"

    return light

def trimeshfile(file, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="trimeshfile Uploading Starting")
    trimesh_file_paths = file.name

    for i in progress.tqdm(trimesh_file_paths, desc="trimeshfile Uploading from file"):
        #sleep(0.1)
        progress(1)

    global trimesh
    trimesh = trimesh_file_paths

    return trimesh_file_paths

def wrpplyfile(file, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="wrpplyfile Uploading Starting")
    wrp_file_paths = file.name

    for i in progress.tqdm(wrp_file_paths, desc="wrpplyfile Uploading from file"):
        #sleep(0.1)
        progress(1)

    global wrpply
    wrpply = wrp_file_paths

    return wrp_file_paths

def pointXmlParing(file):
    point_file_paths = file.name
    tree = parse(point_file_paths)
    root = tree.getroot()
	
    global PointConfig
    PointConfig = point_file_paths
    point_ScaleX = root.find("ScaleX").text
    point_ScaleY = root.find("ScaleY").text
    point_ScaleZ = root.find("ScaleZ").text
    point_Distance = root.find("Distance").text
    point_SLM_PixelPitchX = root.find("SLM_PixelPitchX").text
    point_SLM_PixelPitchY = root.find("SLM_PixelPitchY").text
    point_SLM_PixelNumX = root.find("SLM_PixelNumX").text
    point_SLM_PixelNumY = root.find("SLM_PixelNumY").text
    point_SLM_WaveLength_1 = root.find("SLM_WaveLength_1").text
    point_SLM_WaveLength_2 = root.find("SLM_WaveLength_2").text
    point_SLM_WaveLength_3 = root.find("SLM_WaveLength_3").text

    return {point_Scale_X:point_ScaleX,
            point_Scale_Y:point_ScaleY,
            point_Scale_Z:point_ScaleZ,
            point_Distance_pre:point_Distance,
            point_PixelPitch_X:point_SLM_PixelPitchX,
            point_PixelPitch_Y:point_SLM_PixelPitchY,
            point_PixelNum_X:point_SLM_PixelNumX,
            point_PixelNum_Y:point_SLM_PixelNumY,
            point_WaveLength_1:point_SLM_WaveLength_1,
            point_WaveLength_2:point_SLM_WaveLength_2,
            point_WaveLength_3:point_SLM_WaveLength_3
            }

def depthXmlParing(file):
    depth_file_paths = file.name
    tree = parse(depth_file_paths)
    root = tree.getroot()
    
    global DepthConfig
    DepthConfig = depth_file_paths

    depth_Field_Lens = root.find("FieldLength").text
    depth_Near_Depth = root.find("NearOfDepth").text
    depth_Far_Depth = root.find("FarOfDepth").text
    depth_Num_OfDepth = "256"
    depth_PixelPitch_X= root.find("SLM_PixelPitchX").text
    depth_PixelPitch_Y = root.find("SLM_PixelPitchY").text
    depth_PixelNum_X = root.find("SLM_PixelNumX").text
    depth_PixelNum_Y = root.find("SLM_PixelNumY").text
    depth_Wave_Length1 = root.find("SLM_WaveLength_1").text
    depth_Wave_Length2 = root.find("SLM_WaveLength_2").text
    depth_Wave_Length3 = root.find("SLM_WaveLength_3").text

    return {depth_FieldLength:depth_Field_Lens,
            depth_NearOfDepth:depth_Near_Depth,
            depth_FarOfDepth:depth_Far_Depth,
            depth_NumOfDepth:depth_Num_OfDepth,
            depth_PixelPitchX:depth_PixelPitch_X,
            depth_PixelPitchY:depth_PixelPitch_Y,
            depth_PixelNumX:depth_PixelNum_X,
            depth_PixelNumY:depth_PixelNum_Y,
            depth_WaveLength1:depth_Wave_Length1,
            depth_WaveLength2:depth_Wave_Length2,
            depth_WaveLength3:depth_Wave_Length3
            }


def lightXmlParing(file):
    light_file_paths = file.name
    tree = parse(light_file_paths)
    root = tree.getroot()
   
    global LightConfig
    LightConfig = light_file_paths

    light_distance = root.find("Distance").text
    light_Image_NumOf_X = root.find("Image_NumOfX").text
    light_Image_NumOf_Y = root.find("Image_NumOfY").text
    light_PixelPitch_X= root.find("SLM_PixelPitchX").text
    light_PixelPitch_Y = root.find("SLM_PixelPitchY").text
    light_PixelNum_X = root.find("SLM_PixelNumX").text
    light_PixelNum_Y = root.find("SLM_PixelNumY").text
    light_Wave_Length1 = root.find("SLM_WaveLength_1").text
    light_Wave_Length2 = root.find("SLM_WaveLength_2").text
    light_Wave_Length3 = root.find("SLM_WaveLength_3").text

    return {light_Distance:light_distance,
            light_Image_NumOfX:light_Image_NumOf_X,
            light_Image_NumOfY:light_Image_NumOf_Y,
            light_PixelPitchX:light_PixelPitch_X,
            light_PixelPitchY:light_PixelPitch_Y,
            light_PixelNumX:light_PixelNum_X,
            light_PixelNumY:light_PixelNum_Y,
            light_WaveLength1:light_Wave_Length1,
            light_WaveLength2:light_Wave_Length2,
            light_WaveLength3:light_Wave_Length3
            }

def trimeshXmlParing(file):
    trimesh_file_paths = file.name
    tree = parse(trimesh_file_paths)
    root = tree.getroot()
    
    global TrimeshConfig
    TrimeshConfig = trimesh_file_paths

    trimesh_ScaleX = root.find("ScaleX").text
    trimesh_ScaleY = root.find("ScaleY").text
    trimesh_ScaleZ = root.find("ScaleZ").text
    trimesh_LampDirectionX = root.find("LampDirectionX").text
    trimesh_LampDirectionY = root.find("LampDirectionY").text
    trimesh_LampDirectionZ = root.find("LampDirectionZ").text
    trimesh_PixelPitchX = root.find("SLM_PixelPitchX").text
    trimesh_PixelPitchY = root.find("SLM_PixelPitchY").text
    trimesh_PixelNumX = root.find("SLM_PixelNumX").text
    trimesh_PixelNumY = root.find("SLM_PixelNumY").text
    trimesh_Wave_Length1 = root.find("SLM_WaveLength_1").text
    trimesh_Wave_Length2 = root.find("SLM_WaveLength_2").text
    trimesh_Wave_Length3 = root.find("SLM_WaveLength_3").text

    return {trimesh_Scale_X:trimesh_ScaleX,
            trimesh_Scale_Y:trimesh_ScaleY,
            trimesh_Scale_Z:trimesh_ScaleZ,
            trimesh_LampDirection_X:trimesh_LampDirectionX,
            trimesh_LampDirection_Y:trimesh_LampDirectionY,
            trimesh_LampDirection_Z:trimesh_LampDirectionZ,
            trimesh_PixelPitch_X:trimesh_PixelPitchX,
            trimesh_PixelPitch_Y:trimesh_PixelPitchY,
            trimesh_PixelNum_X:trimesh_PixelNumX,
            trimesh_PixelNum_Y:trimesh_PixelNumY,
            trimesh_WaveLength_1:trimesh_Wave_Length1,
            trimesh_WaveLength_2:trimesh_Wave_Length2,
            trimesh_WaveLength_3:trimesh_Wave_Length3
            }

def wrpXmlParing(file):
    wrp_file_paths = file.name
    tree = parse(wrp_file_paths)
    root = tree.getroot()

    global WrpConfig
    WrpConfig = wrp_file_paths

    print("[wrpXmlParing] root = ",root)
    wrp_ScaleX = root.find("ScaleX").text
    wrp_ScaleY = root.find("ScaleY").text
    wrp_ScaleZ = root.find("ScaleZ").text
    wrp_distance = root.find("Distance").text
    wrp_NumOfWRP = root.find("NumOfWRP").text
    wrp_LocationOfWRP = root.find("LocationOfWRP").text
    wrp_SLM_PixelPitchX = root.find("SLM_PixelPitchX").text
    wrp_SLM_PixelPitchY = root.find("SLM_PixelPitchY").text
    wrp_SLM_PixelNumX = root.find("SLM_PixelNumX").text
    wrp_SLM_PixelNumY = root.find("SLM_PixelNumY").text
    wrp_SLM_WaveLength_1 = root.find("SLM_WaveLength_1").text
    wrp_SLM_WaveLength_2 = root.find("SLM_WaveLength_2").text
    wrp_SLM_WaveLength_3 = root.find("SLM_WaveLength_3").text

    return {wrp_Scale_X:wrp_ScaleX,
            wrp_Scale_Y:wrp_ScaleY,
            wrp_Scale_Z:wrp_ScaleZ,
            wrp_Distance:wrp_distance,
            wrp_NumOf_WRP:wrp_NumOfWRP,
            wrp_WRP_Location:wrp_LocationOfWRP,
            wrp_PixelPitch_X:wrp_SLM_PixelPitchX,
            wrp_PixelPitch_Y:wrp_SLM_PixelPitchY,
            wrp_PixelNum_X:wrp_SLM_PixelNumX,
            wrp_PixelNum_Y:wrp_SLM_PixelNumY,
            wrp_WaveLength_1:wrp_SLM_WaveLength_1,
            wrp_WaveLength_2:wrp_SLM_WaveLength_2,
            wrp_WaveLength_3:wrp_SLM_WaveLength_3
            }

def pointGenerate(Diffraction,Encoding,Precision,pc_mode):
    file = pathlib.Path("OpenholoGeneration")
    plyfile = pathlib.Path(ply)

    if Diffraction == "R-S":
        diff = "0"
    elif Diffraction == "Fresnel":
        diff = "1"

    if Encoding == "Phase":
        enc = "0"
    elif Encoding == "Amplitude":
        enc = "1"
    elif Encoding == "Real":
        enc = "2"
    elif Encoding == "Imaginary":
        enc = "3"

    if Precision == "single":
        pre = "0"
    else:
        pre = "1"

    if pc_mode == "CPU":
        mode = "0"
    else:
        mode = "1"

    if file.exists ():
        if plyfile.exists () :
            subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\" -e \"{}\" -m \"{}\" -p \"{}\" -f \"{}\"'.format("0",PointConfig, ply, enc, pre, mode, diff)], shell=True)
        else:
            print("File upload did not complete. Please try again after the file is uploaded.")
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")
    return './Result/PointCloud_'+Diffraction+"_"+Encoding+'.bmp'

def depthGenerate(Encoding,pc_mode):
    file = pathlib.Path("OpenholoGeneration")
    currentPath = os.getcwd()
    rgbfile = pathlib.Path(currentPath+"//DepthMap//"+rgb+".bmp")
    depthfile = pathlib.Path(currentPath+"//DepthMap//"+depth+".bmp")

    if Encoding == "Phase":
        enc = "0"
    elif Encoding == "Amplitude":
        enc = "1"
    elif Encoding == "Real":
        enc = "2"
    elif Encoding == "Imaginary":
        enc = "3"

    if pc_mode == "CPU":
        mode = "0"
    elif pc_mode == "GPU":
        mode = "1"

    if file.exists () :
        if rgbfile.exists () and depthfile.exists () :
            subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\" \"{}\" \"{}\" -e \"{}\" -m \"{}\" -p \"{}\"'.format("1",DepthConfig, './DepthMap', rgb, depth, enc, "1", mode)], shell=True)
        else:
             print("File upload did not complete. Please try again after the file is uploaded.")
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")

    return './Result/DepthMap_'+Encoding+'.bmp'

def lightGenerate(Encoding,pc_mode):
    file = pathlib.Path("OpenholoGeneration")
    lightdir = pathlib.Path(light)

    if Encoding == "Phase":
        enc = "0"
    elif Encoding == "Amplitude":
        enc = "1"
    elif Encoding == "Real":
        enc = "2"
    elif Encoding == "Imaginary":
        enc = "3"
    if pc_mode == "CPU":
        mode = "0"
    else:
        mode = "1"

    if file.exists ():
        if lightdir.exists () :
            subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\" -e \"{}\" -m \"{}\" -p \"{}\"'.format("2",LightConfig, light, enc, "1", mode)], shell=True)
        else:
            print("File upload did not complete. Please try again after the file is uploaded.")
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")
    return './Result/LightField_'+Encoding+'.bmp'

def trimeshGenerate(Encoding,pc_mode):
    file = pathlib.Path("OpenholoGeneration")
    trimeshfile = pathlib.Path(trimesh)

    if Encoding == "Phase":
        enc = "0"
    elif Encoding == "Amplitude":
        enc = "1"
    elif Encoding == "Real":
        enc = "2"
    elif Encoding == "Imaginary":
        enc = "3"
    if pc_mode == "CPU":
        mode = "0"
    else:
        mode = "1"

    if file.exists ():
        if trimeshfile.exists () :
            subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\" -e \"{}\" -m \"{}\" -p \"{}\"'.format("3",TrimeshConfig, trimesh, enc, "1", mode)], shell=True)
        else:
            print("File upload did not complete. Please try again after the file is uploaded.")
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")
    return './Result/TriMesh_'+Encoding+'.bmp'

def wrpGenerate(Encoding,pc_mode):
    file = pathlib.Path("OpenholoGeneration")
    wrpplyfile = pathlib.Path(wrpply)

    if Encoding == "Phase":
        enc = "0"
    elif Encoding == "Amplitude":
        enc = "1"
    elif Encoding == "Real":
        enc = "2"
    elif Encoding == "Imaginary":
        enc = "3"

    if pc_mode == "CPU":
        mode = "0"
    else:
        mode = "1"

    if file.exists ():
        if wrpplyfile.exists () :
            subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\" -e \"{}\" -m \"{}\" -p \"{}\"'.format("4",WrpConfig, wrpply, enc, "1", mode)], shell=True)
        else:
            print("File upload did not complete. Please try again after the file is uploaded.")
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")
    return './Result/WRP_'+Encoding+'.bmp'

#theme=gr.themes.Soft(),
# gr.themes.Base()/gr.themes.Default()/gr.themes.Glass()/gr.themes.Monochrome()/gr.themes.Soft()    //primary_hue="red", secondary_hue="pink"
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background-color: #CFE3EA}", title="Openholo Generation") as demo:

    gr.HTML("<img src='/file/logo_m.png' align='left' vspace='14' hspace='14'> <font size ='6'><b>Hologram generation methods by Openholo library</b>")

    with gr.Tab(label = "Point Cloud"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    pointxml = gr.File(label="Choose a PointCloud Config (.xml) File to Upload",interactive="True")
                with gr.Column():
                    ply = gr.File(label="Choose a PointCloud (*.ply) File to Upload",interactive="True")
                    plyout = gr.Textbox(label="[Warning] Please wait until the size of the ply file to be uploaded is large and displayed in the temporary path in the text box below. The upload is complete when the box below is displayed.")
            with gr.Row() as row:
                with gr.Column():
                    point_Scale_X = gr.Textbox(label = "Scale X", placeholder="0.01")
                    point_Scale_X.change(lambda s: s, inputs=point_Scale_X)
                with gr.Column():    
                    point_Scale_Y = gr.Textbox(label = "Scale Y", placeholder="0.01")
                    point_Scale_Y.change(lambda s: s, inputs=point_Scale_Y)
                with gr.Column():
                    point_Scale_Z = gr.Textbox(label = "Scale Z", placeholder="0.01")
                    point_Scale_Z.change(lambda s: s, inputs=point_Scale_Z)
            with gr.Row() as row:
                    point_Distance_pre = gr.Textbox(label = "Distance", placeholder="0.5")
                    point_Distance_pre.change(lambda s: s, inputs=point_Distance_pre)
            with gr.Row() as row:
                with gr.Column():
                    point_PixelPitch_X = gr.Textbox(label = "PixelPitch X", placeholder="10e-6")
                    point_PixelPitch_X.change(lambda s: s, inputs=point_PixelPitch_X)
                with gr.Column():
                    point_PixelPitch_Y = gr.Textbox(label = "PixelPitch Y", placeholder="10e-6")
                    point_PixelPitch_Y.change(lambda s: s, inputs=point_PixelPitch_Y)
            with gr.Row() as row:
                with gr.Column():
                    point_PixelNum_X = gr.Textbox(label = "PixelNum X", placeholder="1920")
                    point_PixelNum_X.change(lambda s: s, inputs=point_PixelNum_X)
                with gr.Column():
                    point_PixelNum_Y = gr.Textbox(label = "PixelNum Y", placeholder="1080")
                    point_PixelNum_Y.change(lambda s: s, inputs=point_PixelNum_Y)
            with gr.Row() as row:
                with gr.Column():
                    point_WaveLength_1 = gr.Textbox(label = "WaveLength1", placeholder="638e-9")
                    point_WaveLength_1.change(lambda s: s, inputs=point_WaveLength_1)
                with gr.Column():
                    point_WaveLength_2 = gr.Textbox(label = "WaveLength2", placeholder="520e-9")
                    point_WaveLength_2.change(lambda s: s, inputs=point_WaveLength_2)
                with gr.Column():
                    point_WaveLength_3 = gr.Textbox(label = "WaveLength3", placeholder="450e-9")
                    point_WaveLength_3.change(lambda s: s, inputs=point_WaveLength_3)
            with gr.Row() as row:
                with gr.Column():
                    Diffraction = gr.Dropdown(['R-S','Fresnel'], value="R-S", label = "Diffraction Flag" , info="Choose an flag!")
                with gr.Column():
                    Encoding = gr.Dropdown(['Phase','Amplitude','Real','Imaginary'], value="Phase", label = "Encoding Method" , info="Choose an encoding!")
            with gr.Row() as row:
                with gr.Column():
                    Precision = gr.Radio(["single", "double"], value="single", label="Precision (GPGPU)")
                with gr.Column():
                    pc_mode = gr.Radio(["CPU", "GPU"], value="GPU", label="CPU / GPU")

        pointOutput = gr.Image(label="Point Cloud Bmp")
        pointGen = gr.Button(value = "Generate Point Cloud Hologram")
        pointxml.upload(fn=pointXmlParing, inputs=pointxml,outputs=[point_Scale_X,point_Scale_Y,point_Scale_Z,point_Distance_pre,point_PixelPitch_X,point_PixelPitch_Y,point_PixelNum_X,point_PixelNum_Y,point_WaveLength_1,point_WaveLength_2,point_WaveLength_3])
        ply.upload(fn=plyfile, inputs=ply, outputs=plyout, show_progress="True")
        pointGen.click(fn=pointGenerate,inputs=[Diffraction,Encoding,Precision,pc_mode],outputs=pointOutput)

    with gr.Tab(label = "DepthMap"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    depthxml = gr.File(label="Choose a Depthmap Xml File to Upload",interactive="True")
                with gr.Column():
                    bmprgb = gr.File(label="Choose a Rgb File to Upload",interactive="True")
                    rgbout = gr.Textbox(label="[Warning] Please wait until the size of the rgb file to be uploaded is large and displayed in the temporary path in the text box below. The upload is complete when the box below is displayed.")
                with gr.Column():
                    bmpdepth = gr.File(label="Choose a Depthmap File to Upload",interactive="True")
                    depthout = gr.Textbox(label="[Warning] Please wait until the size of the depth file to be uploaded is large and displayed in the temporary path in the text box below. The upload is complete when the box below is displayed.")
            with gr.Row() as row:
                with gr.Column():
                    depth_FieldLength = gr.Textbox(label = "Field Lens(m)", placeholder="1000e-3")
                    depth_FieldLength.change(lambda s: s, inputs=depth_FieldLength)
                with gr.Column():
                    depth_NumOfDepth = gr.Textbox(label = "Num Of Depth", placeholder="256")
                    depth_NumOfDepth.change(lambda s: s, inputs=depth_NumOfDepth)
            with gr.Row() as row:
                with gr.Column():
                    depth_NearOfDepth = gr.Textbox(label = "Near Depth(m)", placeholder="0.4")
                    depth_NearOfDepth.change(lambda s: s, inputs=depth_NearOfDepth)
                with gr.Column():
                    depth_FarOfDepth = gr.Textbox(label = "Far Depth(m)", placeholder="0.6")
                    depth_FarOfDepth.change(lambda s: s, inputs=depth_FarOfDepth)
            with gr.Row() as row:
                with gr.Column():
                    depth_PixelPitchX = gr.Textbox(label = "PixelPitch X(m)", placeholder="10e-6")
                    depth_PixelPitchX.change(lambda s: s, inputs=depth_PixelPitchX)
                with gr.Column():
                    depth_PixelPitchY = gr.Textbox(label = "PixelPitch Y(m)", placeholder="10e-6")
                    depth_PixelPitchY.change(lambda s: s, inputs=depth_PixelPitchY)
            with gr.Row() as row:
                with gr.Column():
                    depth_PixelNumX = gr.Textbox(label = "PixelNum X", placeholder="1920")
                    depth_PixelNumX.change(lambda s: s, inputs=depth_PixelNumX)
                with gr.Column():
                    depth_PixelNumY = gr.Textbox(label = "PixelNum Y", placeholder="1080")
                    depth_PixelNumY.change(lambda s: s, inputs=depth_PixelNumY)
            with gr.Row() as row:
                with gr.Column():
                    depth_WaveLength1 = gr.Textbox(label = "WaveLength1(m)", placeholder="638e-9")
                    depth_WaveLength1.change(lambda s: s, inputs=depth_WaveLength1)
                with gr.Column():
                    depth_WaveLength2 = gr.Textbox(label = "WaveLength2(m)", placeholder="520e-9")
                    depth_WaveLength2.change(lambda s: s, inputs=depth_WaveLength2)
                with gr.Column():
                    depth_WaveLength3 = gr.Textbox(label = "WaveLength3(m)", placeholder="450e-9")
                    depth_WaveLength3.change(lambda s: s, inputs=depth_WaveLength3)
            with gr.Row() as row:
                with gr.Column():
                    Propagation = gr.Dropdown(['Angular Spectrum'], value="Angular Spectrum", label = "Propagation Method" , info="Choose an Propagation!")
                with gr.Column():
                    Encoding = gr.Dropdown(['Phase','Amplitude','Real','Imaginary'], value="Phase", label = "Encoding Method", info="Choose an Encoding!")
            with gr.Row() as row:
                with gr.Column():
                    pc_mode = gr.Radio(["CPU", "GPU"], value="GPU", label="CPU / GPU", info="Choose an mode!")
        depthxml.upload(fn=depthXmlParing, inputs=depthxml,outputs=[depth_FieldLength,depth_NearOfDepth,depth_FarOfDepth,depth_NumOfDepth,depth_PixelPitchX,depth_PixelPitchY,depth_PixelNumX,depth_PixelNumY,depth_WaveLength1,depth_WaveLength2,depth_WaveLength3])
        bmprgb.upload(fn=rgbfile, inputs=bmprgb, outputs=rgbout, show_progress="True")
        bmpdepth.upload(fn=depthfile, inputs=bmpdepth, outputs=depthout, show_progress="True")
        depthOutput = gr.Image(label="DepthMap Bmp")
        depthGen = gr.Button(value = "Generate DepthMap Hologram")
        depthGen.click(fn=depthGenerate, inputs=[Encoding,pc_mode] ,outputs=depthOutput)

    with gr.Tab(label = "LightField"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    lightxml = gr.File(label="Choose a LightField Xml File to Upload",interactive="True")
                with gr.Column():
                    lightbmp = gr.File(label="Image Folder to Upload",interactive="True", file_count="directory")
            with gr.Row() as row:
                with gr.Column():
                    light_Distance = gr.Textbox(label = "Distance", placeholder="0.5")
                    light_Distance.change(lambda s: s, inputs=light_Distance)
            with gr.Row() as row:
                with gr.Column():
                    light_Image_NumOfX = gr.Textbox(label = "Num of Img X", placeholder="10")
                    light_Image_NumOfX.change(lambda s: s, inputs=light_Image_NumOfX)
                with gr.Column():
                    light_Image_NumOfY = gr.Textbox(label = "Num of Img Y", placeholder="10")
                    light_Image_NumOfY.change(lambda s: s, inputs=light_Image_NumOfY)
            with gr.Row() as row:
                with gr.Column():
                    light_PixelPitchX = gr.Textbox(label = "PixelPitch X(m)", placeholder="1e-5")
                    light_PixelPitchX.change(lambda s: s, inputs=light_PixelPitchX)
                with gr.Column():
                    light_PixelPitchY = gr.Textbox(label = "PixelPitch Y(m)", placeholder="1e-5")
                    light_PixelPitchY.change(lambda s: s, inputs=light_PixelPitchY)
            with gr.Row() as row:
                with gr.Column():
                    light_PixelNumX = gr.Textbox(label = "PixelNum X", placeholder="1920")
                    light_PixelNumX.change(lambda s: s, inputs=light_PixelNumX)
                with gr.Column():
                    light_PixelNumY = gr.Textbox(label = "PixelNum Y", placeholder="1080")
                    light_PixelNumY.change(lambda s: s, inputs=light_PixelNumY)
            with gr.Row() as row:
                with gr.Column():
                    light_WaveLength1 = gr.Textbox(label = "WaveLength1(m)", placeholder="638e-9")
                    light_WaveLength1.change(lambda s: s, inputs=light_WaveLength1)
                with gr.Column():
                    light_WaveLength2 = gr.Textbox(label = "WaveLength2(m)", placeholder="520e-9")
                    light_WaveLength2.change(lambda s: s, inputs=light_WaveLength2)
                with gr.Column():
                    light_WaveLength3 = gr.Textbox(label = "WaveLength3(m)", placeholder="450e-9")
                    light_WaveLength3.change(lambda s: s, inputs=light_WaveLength3)
            with gr.Row() as row:
                with gr.Column():
                    Encoding = gr.Dropdown(['Phase','Amplitude','Real','Imaginary'], value="Phase", label = "Encoding Method", info="Choose an Encoding!")
                with gr.Column():
                    pc_mode = gr.Radio(["CPU", "GPU"], value="GPU", label="CPU / GPU", info="Choose an mode!")
       
        lightxml.upload(fn=lightXmlParing, inputs=lightxml,outputs=[light_Distance,light_Image_NumOfX,light_Image_NumOfY,light_PixelPitchX,light_PixelPitchY,light_PixelNumX,light_PixelNumY,light_WaveLength1,light_WaveLength2,light_WaveLength3])
        lightbmp.upload(fn=lightfile, inputs=lightbmp, show_progress="True")
        lightOutput = gr.Image(label="LightField Bmp")
        lightGen = gr.Button(value = "Generate LightField Hologram")
        lightGen.click(fn=lightGenerate, inputs=[Encoding,pc_mode] ,outputs=lightOutput)

    with gr.Tab(label = "Triangle Mesh"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    trimeshxml = gr.File(label="Choose a Triangle Mesh Config (.xml) File to Upload",interactive="True")
                with gr.Column():
                    trimeshply = gr.File(label="Choose a Triangle Mesh (*.ply) File to Upload",interactive="True")
                    trimeshplyout = gr.Textbox(label="[Warning] Please wait until the size of the Triangle Mesh file to be uploaded is large and displayed in the temporary path in the text box below. The upload is complete when the box below is displayed.")
            with gr.Row() as row:
                with gr.Column():
                    trimesh_Scale_X = gr.Textbox(label = "Scale X", placeholder="0.005")
                    trimesh_Scale_X.change(lambda s: s, inputs=trimesh_Scale_X)
                with gr.Column():
                    trimesh_Scale_Y = gr.Textbox(label = "Scale Y", placeholder="0.005")
                    trimesh_Scale_Y.change(lambda s: s, inputs=trimesh_Scale_Y)
                with gr.Column():
                    trimesh_Scale_Z = gr.Textbox(label = "Scale Z", placeholder="0.005")
                    trimesh_Scale_Z.change(lambda s: s, inputs=trimesh_Scale_Z)
            with gr.Row() as row:
                with gr.Column():
                    trimesh_LampDirection_X = gr.Textbox(label = "LampDirectionX", placeholder="0")
                    trimesh_LampDirection_X.change(lambda s: s, inputs=trimesh_LampDirection_X) 
                with gr.Column():
                    trimesh_LampDirection_Y = gr.Textbox(label = "LampDirectionY", placeholder="0.3")
                    trimesh_LampDirection_Y.change(lambda s: s, inputs=trimesh_LampDirection_Y)
                with gr.Column():
                    trimesh_LampDirection_Z = gr.Textbox(label = "LampDirectionZ", placeholder="1")
                    trimesh_LampDirection_Z.change(lambda s: s, inputs=trimesh_LampDirection_Z)
            with gr.Row() as row:
                with gr.Column():
                    trimesh_PixelPitch_X = gr.Textbox(label = "PixelPitch X", placeholder="10e-6")
                    trimesh_PixelPitch_X.change(lambda s: s, inputs=trimesh_PixelPitch_X)
                with gr.Column():
                    trimesh_PixelPitch_Y = gr.Textbox(label = "PixelPitch Y", placeholder="10e-6")
                    trimesh_PixelPitch_Y.change(lambda s: s, inputs=trimesh_PixelPitch_Y)
            with gr.Row() as row:
                with gr.Column():
                    trimesh_PixelNum_X = gr.Textbox(label = "PixelNum X", placeholder="1920")
                    trimesh_PixelNum_X.change(lambda s: s, inputs=trimesh_PixelNum_X)
                with gr.Column():
                    trimesh_PixelNum_Y = gr.Textbox(label = "PixelNum Y", placeholder="1080")
                    trimesh_PixelNum_Y.change(lambda s: s, inputs=trimesh_PixelNum_Y)
            with gr.Row() as row:
                with gr.Column():
                    trimesh_WaveLength_1 = gr.Textbox(label = "WaveLength1", placeholder="638e-9")
                    trimesh_WaveLength_1.change(lambda s: s, inputs=trimesh_WaveLength_1)
                with gr.Column():
                    trimesh_WaveLength_2 = gr.Textbox(label = "WaveLength2", placeholder="520e-9")
                    trimesh_WaveLength_2.change(lambda s: s, inputs=trimesh_WaveLength_2)
                with gr.Column():
                    trimesh_WaveLength_3 = gr.Textbox(label = "WaveLength3", placeholder="450e-9")
                    trimesh_WaveLength_3.change(lambda s: s, inputs=trimesh_WaveLength_3)
            with gr.Row() as row:
                with gr.Column():
                    Encoding = gr.Dropdown(['Phase','Amplitude','Real','Imaginary'], value="Phase", label = "Encoding Method" , info="Choose an encoding!")
                with gr.Column():
                    pc_mode = gr.Radio(["CPU", "GPU"], value="GPU", label="CPU / GPU")
        trimeshOutput = gr.Image(label="Triangle Mesh Bmp")
        trimeshGen = gr.Button(value = "Generate Triangle Mesh Hologram")
        trimeshxml.upload(fn=trimeshXmlParing, inputs=trimeshxml,outputs=[trimesh_Scale_X,trimesh_Scale_Y,trimesh_Scale_Z,trimesh_LampDirection_X,trimesh_LampDirection_Y,trimesh_LampDirection_Z,trimesh_PixelPitch_X,trimesh_PixelPitch_Y,trimesh_PixelNum_X,trimesh_PixelNum_Y,trimesh_WaveLength_1,trimesh_WaveLength_2,trimesh_WaveLength_3])
        trimeshply.upload(fn=trimeshfile, inputs=trimeshply, outputs=trimeshplyout, show_progress="True")
        trimeshGen.click(fn=trimeshGenerate,inputs=[Encoding,pc_mode],outputs=trimeshOutput)
    
    with gr.Tab(label = "WRP"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    wrpxml = gr.File(label="Choose a WRP Config (.xml) File to Upload",interactive="True")
                with gr.Column():
                    wrpply = gr.File(label="Choose a WRP (*.ply) File to Upload",interactive="True")
                    wrpplyout = gr.Textbox(label="[Warning] Please wait until the size of the ply file to be uploaded is large and displayed in the temporary path in the text box below. The upload is complete when the box below is displayed.")
            with gr.Row() as row:
                with gr.Column():
                    wrp_Scale_X = gr.Textbox(label = "Scale X", placeholder="0.0015")
                    wrp_Scale_X.change(lambda s: s, inputs=wrp_Scale_X)
                with gr.Column():
                    wrp_Scale_Y = gr.Textbox(label = "Scale Y", placeholder="0.0015")
                    wrp_Scale_Y.change(lambda s: s, inputs=wrp_Scale_Y)
                with gr.Column():
                    wrp_Scale_Z = gr.Textbox(label = "Scale Z", placeholder="0.0015")
                    wrp_Scale_Z.change(lambda s: s, inputs=wrp_Scale_Z)
            with gr.Row() as row:
                with gr.Column():
                    wrp_Distance = gr.Textbox(label = "Distance", placeholder="0.5")
                    wrp_Distance.change(lambda s: s, inputs=wrp_Distance)
                with gr.Column():
                    wrp_NumOf_WRP = gr.Textbox(label = "NumOfWRP", placeholder="1")
                    wrp_NumOf_WRP.change(lambda s: s, inputs=wrp_NumOf_WRP)
                with gr.Column():
                    wrp_WRP_Location = gr.Textbox(label = "LocationOfWRP", placeholder="0.003")
                    wrp_WRP_Location.change(lambda s: s, inputs=wrp_WRP_Location)
            with gr.Row() as row:
                with gr.Column():
                    wrp_PixelPitch_X = gr.Textbox(label = "PixelPitch X", placeholder="1e-5")
                    wrp_PixelPitch_X.change(lambda s: s, inputs=wrp_PixelPitch_X)
                with gr.Column():
                    wrp_PixelPitch_Y = gr.Textbox(label = "PixelPitch Y", placeholder="1e-5")
                    wrp_PixelPitch_Y.change(lambda s: s, inputs=wrp_PixelPitch_Y)
            with gr.Row() as row:
                with gr.Column():
                    wrp_PixelNum_X = gr.Textbox(label = "PixelNum X", placeholder="1920")
                    wrp_PixelNum_X.change(lambda s: s, inputs=wrp_PixelNum_X)
                with gr.Column():
                    wrp_PixelNum_Y = gr.Textbox(label = "PixelNum Y", placeholder="1080")
                    wrp_PixelNum_Y.change(lambda s: s, inputs=wrp_PixelNum_Y)
            with gr.Row() as row:
                with gr.Column():
                    wrp_WaveLength_1 = gr.Textbox(label = "WaveLength1", placeholder="638e-9")
                    wrp_WaveLength_1.change(lambda s: s, inputs=wrp_WaveLength_1)
                with gr.Column():
                    wrp_WaveLength_2 = gr.Textbox(label = "WaveLength2", placeholder="520e-9")
                    wrp_WaveLength_2.change(lambda s: s, inputs=wrp_WaveLength_2)
                with gr.Column():
                    wrp_WaveLength_3 = gr.Textbox(label = "WaveLength3", placeholder="450e-9")
                    wrp_WaveLength_3.change(lambda s: s, inputs=wrp_WaveLength_3)
            with gr.Row() as row:
                with gr.Column():
                    Encoding = gr.Dropdown(['Phase','Amplitude','Real','Imaginary'], value="Phase", label = "Encoding Method" , info="Choose an encoding!")
                with gr.Column():
                    pc_mode = gr.Radio(["CPU", "GPU"], value="GPU", label="CPU / GPU")
        wrpOutput = gr.Image(label="WRP Bmp")
        wrpGen = gr.Button(value = "Generate WRP Hologram")
        wrpxml.upload(fn=wrpXmlParing, inputs=wrpxml,outputs=[ wrp_Scale_X, wrp_Scale_Y, wrp_Scale_Z, wrp_Distance, wrp_NumOf_WRP, wrp_WRP_Location, wrp_PixelPitch_X, wrp_PixelPitch_Y, wrp_PixelNum_X, wrp_PixelNum_Y, wrp_WaveLength_1, wrp_WaveLength_2, wrp_WaveLength_3])
        wrpply.upload(fn=wrpplyfile, inputs=wrpply, outputs=wrpplyout, show_progress="True")
        wrpGen.click(fn=wrpGenerate,inputs=[Encoding,pc_mode],outputs=wrpOutput)

if __name__ == "__main__":
    demo.queue(max_size=100).launch(share=True)
    #demo.queue(concurrency_count=10).launch(share=True)
    #demo.launch(share=True)
