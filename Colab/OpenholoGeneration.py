import gradio as gr
import subprocess
import sys
import pathlib
from xml.etree.ElementTree import parse


def pointGenerate(Config, ply):
    file = pathlib.Path("OpenholoGeneration")
    if file.exists ():
        subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\"'.format("0",Config, ply)], shell=True)
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")
    return './Result/PointCloud_phase.bmp'

def pointXmlParing(config):
    tree = parse(config)
    root = tree.getroot()

    FieldLength = root.find("FieldLength").text
    ScaleX = root.find("ScaleX").text
    ScaleY = root.find("ScaleY").text
    ScaleZ = root.find("ScaleZ").text
    Distance = root.find("Distance").text
    SLM_PixelPitchX = root.find("SLM_PixelPitchX").text
    SLM_PixelPitchY = root.find("SLM_PixelPitchY").text
    SLM_PixelNumX = root.find("SLM_PixelNumX").text
    SLM_PixelNumY = root.find("SLM_PixelNumY").text
    SLM_WaveLength_1 = root.find("SLM_WaveLength_1").text
    SLM_WaveLength_2 = root.find("SLM_WaveLength_2").text
    SLM_WaveLength_3 = root.find("SLM_WaveLength_3").text

    return {Field_Length:FieldLength,
            Scale_X:ScaleX,
            Scale_Y:ScaleY,
            Scale_Z:ScaleZ,
            Distance_pre:Distance,
            PixelPitch_X:SLM_PixelPitchX,
            PixelPitch_Y:SLM_PixelPitchY,
            PixelNum_X:SLM_PixelNumX,
            PixelNum_Y:SLM_PixelNumY,
            WaveLength_1:SLM_WaveLength_1,
            WaveLength_2:SLM_WaveLength_2,
            WaveLength_3:SLM_WaveLength_3
            }

def depthGenerate(depthConfig, ImgPath, Rgb, Depth):
    file = pathlib.Path("OpenholoGeneration")
    if file.exists ():
        subprocess.run(['./OpenholoGeneration -a \"{}\" -c \"{}\" -i \"{}\" \"{}\" \"{}\"'.format("1",depthConfig, ImgPath, Rgb, Depth)], shell=True)
    else:
        print("OpenholoGeneration file does not exist, so it does not work normally")
    return './Result/DepthMap_phase.bmp'

def depthXmlParing(config):
    tree = parse(config)
    root = tree.getroot()

    Field_Lens = root.find("FieldLength").text
    Near_Depth = root.find("NearOfDepth").text
    Far_Depth = root.find("FarOfDepth").text
    Num_OfDepth = "256"
    PixelPitch_X= root.find("SLM_PixelPitchX").text
    PixelPitch_Y = root.find("SLM_PixelPitchY").text
    PixelNum_X = root.find("SLM_PixelNumX").text
    PixelNum_Y = root.find("SLM_PixelNumY").text
    Wave_Length1 = root.find("SLM_WaveLength_1").text
    Wave_Length2 = root.find("SLM_WaveLength_2").text
    Wave_Length3 = root.find("SLM_WaveLength_3").text

    return {FieldLength:Field_Lens,
            NearOfDepth:Near_Depth,
            FarOfDepth:Far_Depth,
            NumOfDepth:Num_OfDepth,
            SLM_PixelPitchX:PixelPitch_X,
            SLM_PixelPitchY:PixelPitch_Y,
            SLM_PixelNumX:PixelNum_X,
            SLM_PixelNumY:PixelNum_Y,
            WaveLength1:Wave_Length1,
            WaveLength2:Wave_Length2,
            WaveLength3:Wave_Length3,
            }

#theme=gr.themes.Soft(),
# gr.themes.Base()/gr.themes.Default()/gr.themes.Glass()/gr.themes.Monochrome()/gr.themes.Soft()    //primary_hue="red", secondary_hue="pink"
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background-color: #CFE3EA}", title="Openholo Generation") as demo:

    gr.HTML("<img src='/file/logo_m.png' align='left' vspace='14' hspace='14'> <font size ='6'><b>Hologram generation methods by Openholo library</b>")

    with gr.Tab(label = "Point Cloud"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    Field_Length = gr.Textbox(label = "FieldLength")
                    Field_Length.change(lambda s: s, inputs=Field_Length)
                    Scale_X = gr.Textbox(label = "Scale X")
                    Scale_X.change(lambda s: s, inputs=Scale_X)
                    Scale_Y = gr.Textbox(label = "Scale Y")
                    Scale_Y.change(lambda s: s, inputs=Scale_Y)
                    Scale_Z = gr.Textbox(label = "Scale Z")
                    Scale_Z.change(lambda s: s, inputs=Scale_Z)
                    Distance_pre = gr.Textbox(label = "Distance")
                    Distance_pre.change(lambda s: s, inputs=Distance_pre)
                    PixelPitch_X = gr.Textbox(label = "PixelPitch X")
                    PixelPitch_X.change(lambda s: s, inputs=PixelPitch_X)
                with gr.Column():
                    PixelPitch_Y = gr.Textbox(label = "PixelPitch Y")
                    PixelPitch_Y.change(lambda s: s, inputs=PixelPitch_Y)
                    PixelNum_X = gr.Textbox(label = "PixelNum X")
                    PixelNum_X.change(lambda s: s, inputs=PixelNum_X)
                    PixelNum_Y = gr.Textbox(label = "PixelNum Y")
                    PixelNum_Y.change(lambda s: s, inputs=PixelNum_Y)
                    WaveLength_1 = gr.Textbox(label = "WaveLength1")
                    WaveLength_1.change(lambda s: s, inputs=WaveLength_1)
                    WaveLength_2 = gr.Textbox(label = "WaveLength2")
                    WaveLength_2.change(lambda s: s, inputs=WaveLength_2)
                    WaveLength_3 = gr.Textbox(label = "WaveLength3")
                    WaveLength_3.change(lambda s: s, inputs=WaveLength_3)
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    Config = gr.Dropdown(['./Config/Generation_PointCloud (RGB).xml'], label = "Config (.xml)")
                    pointXml = gr.Button(value = "XmlLoad")
                with gr.Column():
                    ply = gr.Dropdown(['./PointCloud & WRP/pointcloud_1470.ply','./PointCloud & WRP/pointcloud_3748.ply'], label = "PointCloud (*.ply)")
                    pointGen = gr.Button(value = "Generate Point Cloud Hologram")
        pointOutput = gr.Image(label="Point Cloud Bmp")
        pointGen.click(fn=pointGenerate, inputs=[Config, ply], outputs=pointOutput)
        pointXml.click(fn=pointXmlParing, inputs=Config, outputs=[Field_Length,Scale_X,Scale_Y,Scale_Z,Distance_pre,PixelPitch_X,PixelPitch_Y,PixelNum_X,PixelNum_Y,WaveLength_1,WaveLength_2,WaveLength_3])

    with gr.Tab(label = "DepthMap"):
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    FieldLength = gr.Textbox(label = "Field Lens(m)")
                    FieldLength.change(lambda s: s, inputs=FieldLength)
                    NearOfDepth = gr.Textbox(label = "Near Depth(m)")
                    NearOfDepth.change(lambda s: s, inputs=NearOfDepth)
                    FarOfDepth = gr.Textbox(label = "Far Depth(m)")
                    FarOfDepth.change(lambda s: s, inputs=FarOfDepth)
                    NumOfDepth = gr.Textbox(label = "Num Of Depth")
                    NumOfDepth.change(lambda s: s, inputs=NumOfDepth)
                    SLM_PixelPitchX = gr.Textbox(label = "PixelPitch X(m)")
                    SLM_PixelPitchX.change(lambda s: s, inputs=SLM_PixelPitchX)
                    SLM_PixelPitchY = gr.Textbox(label = "PixelPitch Y(m)")
                    SLM_PixelPitchY.change(lambda s: s, inputs=SLM_PixelPitchY)
                    SLM_PixelNumX = gr.Textbox(label = "PixelNum X")
                    SLM_PixelNumX.change(lambda s: s, inputs=SLM_PixelNumX)
                    SLM_PixelNumY = gr.Textbox(label = "PixelNum Y")
                    SLM_PixelNumY.change(lambda s: s, inputs=SLM_PixelNumY)
                    depthConfig = gr.Dropdown(['./Config/Generation_DepthMap (RGB).xml'], label = "Config (.xml)")
                with gr.Column():
                    WaveLength1 = gr.Textbox(label = "WaveLength1(m)")
                    WaveLength1.change(lambda s: s, inputs=WaveLength1)
                    WaveLength2 = gr.Textbox(label = "WaveLength2(m)")
                    WaveLength2.change(lambda s: s, inputs=WaveLength2)
                    WaveLength3 = gr.Textbox(label = "WaveLength3(m)")
                    WaveLength3.change(lambda s: s, inputs=WaveLength3)
                    Propagation = gr.Dropdown(['Angular Spectrum'], label = "Propagation Method")
                    Encoding = gr.Dropdown(['Phase'], label = "Encoding Method")
                    ImgPath = gr.Dropdown(['./DepthMap'], label = "ImgPath")
                    Rgb = gr.Dropdown(['DepthMap_rgb'], label = "RGB (.bmp)")
                    Depth = gr.Dropdown(['DepthMap_256bit'], label = "DepthMap (.bmp)")
        with gr.Column():
            with gr.Row() as row:
                with gr.Column():
                    Xmlbtn = gr.Button(value = "XmlLoad")
                with gr.Column():
                    depthGen = gr.Button(value = "Generate DepthMap Hologram")
        output = gr.Image(label="DepthMap Bmp")
    Xmlbtn.click(fn=depthXmlParing, inputs=depthConfig, outputs=[FieldLength,NearOfDepth,FarOfDepth,NumOfDepth,SLM_PixelPitchX,SLM_PixelPitchY,SLM_PixelNumX,SLM_PixelNumY,WaveLength1,WaveLength2,WaveLength3])
    depthGen.click(fn=depthGenerate, inputs=[depthConfig,ImgPath,Rgb,Depth], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
