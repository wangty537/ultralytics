from time import sleep
from ultralytics import YOLO
import torch


def load_pretrained_weights(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    
    # 1. 先加载所有能直接匹配的权重（非 ECB 部分）
    for name, param in pretrained_dict.items():
        #print(f"Loading {name} from pretrained...")
        if name in model_dict and "conv3x3" not in name and "conv1x1_" not in name:
            #print(f"ddd {name} from pretrained...")
            # 判断shape是否匹配
            if model_dict[name].shape == param.shape:
                # 如果形状匹配，则直接复制权重
                #print(f"Copying {name} from pretrained...")
                model_dict[name].copy_(param)
            #else:
                # 如果形状不匹配，打印警告信息
                #print(f"Shape mismatch for {name}: model {model_dict[name].shape}, pretrained {param.shape}")
    
    # 2. 处理 ECB 模块的 conv3x3 权重
    for name, param in model_dict.items():
       
        if "conv3x3" in name:
            
            # 构造 pretrained 中对应的 conv 名称
            # 例如：model 的 "0.cv1.conv.ECB.conv3x3.weight" 
            # 对应 pretrained 的 "0.cv1.conv.weight"
            pretrained_key = name.replace(".conv3x3", "")
            #print('ecb',name, pretrained_key)
            if pretrained_key in pretrained_dict:
                param.copy_(pretrained_dict[pretrained_key])
    
    # 3. 加载调整后的权重（允许部分不匹配）
    #model.load_state_dict(pretrained_dict, strict=False)
    
if __name__ == "__main__":
    # 加载新模型
    model = YOLO("yolo11m.yaml")
    # 加载pretrain权重
    pretrain = torch.load("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/detect/yolo11m/train640/weights/best.pt")



    # print(model.model.model)
    # print(pretrain["model"].model)
    #sleep(1033333333)
    print("Loading pretrain weights...")
    # 自定义加载
    # model1 = custom_load_state_dict(model.model.model, pretrain["model"].model)
    # print("Custom weight loading done.")

    load_pretrained_weights(model.model.model, pretrain["model"].model)
    # 打印weights加载情况
    print("Pretrained weights loaded successfully.")
    print(model.model.model.state_dict()['23.cv3.2.1.1.conv.weight'])
    print(pretrain["model"].model.state_dict()['23.cv3.2.1.1.conv.weight']) # 23.cv3.2.1.0.conv.conv3x3.weight


    print(model.model.model.state_dict()['8.m.0.m.1.cv2.conv.conv3x3.weight'])
    print(pretrain["model"].model.state_dict()['8.m.0.m.1.cv2.conv.weight'])

    print(model.model.model.state_dict()['23.cv3.2.1.1.conv.weight']==pretrain["model"].model.state_dict()['23.cv3.2.1.1.conv.weight'])
    print(model.model.model.state_dict()['8.m.0.m.1.cv2.conv.conv3x3.weight']==pretrain["model"].model.state_dict()['8.m.0.m.1.cv2.conv.weight'])
    # input = torch.randn(1, 3, 320, 320).cuda()
    # #output1 = model1(input)
    # output2 = model.predict(input)
    # #print("Output1:", output1)
    # print("Output2:", output2)


