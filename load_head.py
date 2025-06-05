from time import sleep
from ultralytics import YOLO
import torch


def load_pretrained_weights_head(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    
    # 1. 先加载所有能直接匹配的权重
    for name, param in pretrained_dict.items():
        print(f"Loading {name} from pretrained...")
        if name in model_dict and "17" not in name and "18" not in name and "19" not in name and "20" not in name and "21" not in name and "22" not in name and "23" not in name \
              and "24" not in name and "25" not in name:
            print(f"ddd {name} from pretrained...")
            model_dict[name].copy_(param)
    

    
    # 3. 加载调整后的权重（允许部分不匹配）
    #model.load_state_dict(pretrained_dict, strict=False)
    return model
if __name__ == "__main__":
    # 加载新模型
    model = YOLO("yolo18m.yaml") # 修改Conv类为原始的
    # 加载pretrain权重
    pretrain = torch.load("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/detect/yolo11m/train640/weights/best.pt")



    # print(model.model.model)
    # print(pretrain["model"].model)
    #sleep(1033333333)
    print("Loading pretrain weights...")
    # 自定义加载
    # model1 = custom_load_state_dict(model.model.model, pretrain["model"].model)
    # print("Custom weight loading done.")

    load_pretrained_weights_head(model.model.model, pretrain["model"].model)
    # 打印weights加载情况
    print("Pretrained weights loaded successfully.")
   


    print(model.model.model.state_dict()['16.m.0.m.1.cv2.conv.weight'])
    print(pretrain["model"].model.state_dict()['16.m.0.m.1.cv2.conv.weight'])

    print(model.model.model.state_dict()['8.m.0.m.1.cv2.conv.weight'])
    print(pretrain["model"].model.state_dict()['8.m.0.m.1.cv2.conv.weight'])

    print(model.model.model.state_dict()['16.m.0.m.1.cv2.conv.weight']==pretrain["model"].model.state_dict()['16.m.0.m.1.cv2.conv.weight'])
    print(model.model.model.state_dict()['8.m.0.m.1.cv2.conv.weight']==pretrain["model"].model.state_dict()['8.m.0.m.1.cv2.conv.weight'])

    # input = torch.randn(1, 3, 320, 320).cuda()
    # #output1 = model1(input)
    # output2 = model.predict(input)
    # #print("Output1:", output1)
    # print("Output2:", output2)


