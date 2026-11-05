#插件
import torch
from torch.utils.data import Dataset,DataLoader
import cv2 
import os
import json
import numpy as np
import subprocess
#输入
class PhiraDataset(Dataset):
    def __init__(self,data_dir,transform=None, chunk_len=16,warmup_len = 4,max_slots = 2, fps = 60.0):
        self.data_dir=data_dir
        self.transform=transform
        self.chunk_len= chunk_len 
        self.warmup_len= warmup_len
        self.max_slots = max_slots
        self.fps = fps
        self.samples = []

        for paths,dirs,files in os.walk(data_dir):
            video_path = None
            chart_path = None
            for file_name in files:
                if file_name.endswith(".mp4"):
                    video_path = os.path.join(paths,file_name)
                elif file_name.endswith(".json"):
                    chart_path = os.path.join(paths,file_name)

            if video_path and chart_path:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS) 
                if video_fps == 0 or video_fps is None:
                    video_fps = 60.0
                cap.release()
                for start_index in range(self.warmup_len,total_frames-self.chunk_len+1):
                    self.samples.append((video_path,chart_path,start_index))
    def __len__(self):
        return len(self.samples)

    def read_rgb(self,video_file,start_frame,num_frames,fps = 60.0):
        start_time = start_frame/fps
        command = [
            'ffmpeg', 
            '-ss', str(start_time),      # 定位到起点时间
            '-i', video_file,              # 输入文件
            '-frames:v', str(num_frames),# 读取的帧数 (warmup + T)
            '-vf', 'scale=320:180',      # ★ 核心：让 ffmpeg 直接缩放，不占 Python 内存，也不需要 self.H/W 变量
            '-f', 'rawvideo', 
            '-pix_fmt', 'rgb24',         # ★ 核心：强制输出 RGB 8bit 三通道，绝不转灰阶
            '-loglevel', 'error', 
            'pipe:1'                     # 将数据输出到管道
        ]
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr = subprocess.PIPE)
        out, err = process.communicate()
        video_array = np.frombuffer(out,dtype = np.unit8)
        video_array = video_array.reshape(-1,180,320,3)
        video_array = video_array.transpose(0,3,1,2)
        return video_array


    
#调用
    def __getitem__(self, index):
        video_path,chart_path ,start_index,= self.samples[index]
        total_frame = self.warmup_len + self.chunk_len
        read_start = start_index - self.warmup_len
        video_np = self.read_rgb(video_path,read_start,total_frame)
        warmup_x = torch.from_numpy(video_np[:self.warmup_len]).float()/255.0
        target_x = torch.from_numpy(video_np[self.warmup_len:]).float()/255.0
        
        with open(chart_path, "r",encoding='utf-8') as file: 
            chart_data = json.load(file)

        # ★ [关键改动 1]：调用你写好的解析函数
        # 假设你的 parse_finger_jsonl 返回的是字典：{ 帧号: [{'type': 1, 'coord': [x,y]}, ...] }
            frame_labels = __(chart_data)

        type_label = torch.zeros((self.chunk_len,self.max_slots),dtype = torch.long)
        coord_label = torch.zeros((self.chunk_len,self.max_slots,2),dtype = torch.float32)
        for t in range (self.chunk_len):
            current_index = start_index + t

            fingers = frame_labels.get(current_index,[])
            for i, finger in enumerate(fingers):
                if i >= self.max_slots:
                   break
                type_label[t,i] = finger['type']
                coord_label[t,i] = torch.tensor(finger['coord'],dtype=torch.float32)
        return  warmup_x,target_x,type_label,coord_label

#测试
if __name__ == "__main__" :

    #此处请替换为谱面在电脑中的实际路径
    dataset = PhiraDataset(data_dir="C:/Users/HUAWEI/OneDrive/Desktop/Mapset/map-set-train/map-set-train.dir")
    
    
    chunk_len = 16
    warmup_len = 4
    max_slots = 2
    fps = 60.0
    
    print(f"共收集{len(dataset)}份谱面")

    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    print("开始读取...")
    for warmup_x,target_x,type_label,coord_label in dataloader:
        print("数据形状")
        print("预热：",warmup_x.shape)
        print("目标：",target_x.shape)
        print("动作类型：",type_label.shape)
        print("动作坐标：",coord_label.shape)
        print("测试1:")
        print("手指0动作类型:",type_label[0,0,0].item())
        print("手指0动作坐标:",coord_label[0,0,0].tolist())
        break
    print("测试成功力！")
    if len(dataset)>0:
        for frames, labels in dataloader:
            if frames is None:
               continue 
            print("视频形状",frames.shape)
            print("标签形状", labels.shape)
            print("前五个音符时间",labels[0][:5].tolist())
            break




        


