"""Compare speed of different models with batch size 12"""
import torch
import torchvision.models as models
import platform,psutil
import torch.nn as nn
import time,os
import pandas as pd
import argparse
import shutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pynvml as nvml

torch.backends.cudnn.benchmark = True

RESULT_ROOT_DIR = './experiment_results'
ENVS_BENCHMARK_IMAGE_SAVE_DIR = './envs_benchmark_result'
GPUS_BENCHMARK_IMAGE_SAVE_DIR = './gpus_benchmark_result'
DF_COLUMNS = ['envs','gpus','models','phases','precisions','time']

MODEL_LIST = {
    models.mnasnet:models.mnasnet.__all__[1:],
    models.resnet: models.resnet.__all__[1:4],
    models.densenet: models.densenet.__all__[1:],
    models.squeezenet: models.squeezenet.__all__[1:],
    models.vgg: models.vgg.__all__[1:],
    models.mobilenet:models.mobilenet.__all__[1:],
    models.shufflenetv2:models.shufflenetv2.__all__[1:]
}


precisions=["float","half",'double']
phases = ['train','inference']

device_name=str(torch.cuda.get_device_name(0))

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')

parser.add_argument('opertation',type=str,choices=['run','gpu','env'],help='opertation must be run gpu or env')
parser.add_argument('--ENVS','-es', type=str, nargs='+', required=True,default=[],help='Name of enviroment')
parser.add_argument('--GPUS','-gs',type=str, nargs='*',required=False,help='Name of gpu')
parser.add_argument('--WARM_UP','-w', type=int,default=5, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=50,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=12, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=1000, required=False, help='Num of class')
parser.add_argument('--NUM_GPU','-g', type=int, default=1, required=False, help='Num of gpus')

args = parser.parse_args()
args.BATCH_SIZE*=args.NUM_GPU

class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn( 3, 224, 224,length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset( args.BATCH_SIZE*(args.WARM_UP + args.NUM_TEST)),
                         batch_size=args.BATCH_SIZE, shuffle=False,num_workers=8)
def train(type='single'):
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if args.NUM_GPU > 1:
                model = nn.DataParallel(model,device_ids=range(args.NUM_GPU))
            model=getattr(model,type)()
            model=model.to('cuda')
            durations = []
            print('Benchmarking Training {} precision type {} '.format(type,model_name))
            for step,img in enumerate(rand_loader):
                img=getattr(img,type)()
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to('cuda'))
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= args.WARM_UP:
                    durations.append((end - start)*1000)
            print(model_name,' model average train time : ',sum(durations)/len(durations),'ms')
            del model
            benchmark[model_name] = durations
    return benchmark

def inference(type='float'):
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model,device_ids=range(args.NUM_GPU))
                model=getattr(model,type)()
                model=model.to('cuda')
                model.eval()
                durations = []
                print('Benchmarking Inference {} precision type {} '.format(type,model_name))
                for step,img in enumerate(rand_loader):
                    img=getattr(img,type)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to('cuda'))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start)*1000)
                print(model_name,' model average inference time : ',sum(durations)/len(durations),'ms')
                del model
                benchmark[model_name] = durations
    return benchmark



def experiment(env_name,device_name):
    
    system_configs=str(platform.uname())
    system_configs='\n'.join((system_configs,str(psutil.cpu_freq()),'cpu_count: '+str(psutil.cpu_count()),'memory_available: '+str(psutil.virtual_memory().available)))

    gpu_configs=[torch.cuda.device_count(),torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0)]
    gpu_configs=list(map(str,gpu_configs))
    temp=['Number of GPUs on current device : ','CUDA Version : ','Cudnn Version : ','Device Name : ']

    experiment_result_data_dir = os.path.join(RESULT_ROOT_DIR,env_name,device_name,'data')
    if os.path.exists(experiment_result_data_dir):
        shutil.rmtree(experiment_result_data_dir)
    os.makedirs(experiment_result_data_dir)

    now = time.localtime()
    start_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark start : ',start_time)


    for idx,value in enumerate(zip(temp,gpu_configs)):
        gpu_configs[idx]=''.join(value)
        print(gpu_configs[idx])
    print(system_configs)



    with open(os.path.join(experiment_result_data_dir,"system_info.txt"), "w") as f:
        f.writelines('benchmark start : '+start_time+'\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs )

    
    for precision in precisions:
        train_result=train(precision)
        train_result_df = pd.DataFrame(train_result)
        path=''.join((experiment_result_data_dir,'/',device_name.replace(" ","_"),"_",precision,'_model_train_benchmark.csv'))
        train_result_df.to_csv(path, index=False)

        print("save {} succeed !".format(path))

        inference_result=inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path=''.join((experiment_result_data_dir,'/',device_name.replace(" ","_"),"_",precision,'_model_inference_benchmark.csv'))
        inference_result_df.to_csv(path, index=False)

        print("save {} succeed !".format(path))

    now = time.localtime()
    end_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark end : ',end_time)
    with open(os.path.join(experiment_result_data_dir,"system_info.txt"), "a") as f:
        f.writelines('benchmark end : '+end_time+'\n')

def get_sub_dir(parent_dir):
    files = os.listdir(parent_dir)
    sub_dirs = []
    for file in files:
        if os.path.isdir(os.path.join(parent_dir,file)):
            sub_dirs.append(file)
    return sub_dirs


def insert_series_to_df(df,env,gpu,phase,precision,average_time):
    for model,time in average_time.iteritems():
        df = df.append({"envs":env,"gpus":gpu,"models":model,"phases":phase,"precisions":precision,"time":int(time)},ignore_index=True)
    return df

def build_small_data_frame_for_benchmark(env,gpu,benchmark_csv_files_dir):
    files = os.listdir(benchmark_csv_files_dir)
    csv_files = []
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(file)

    df = pd.DataFrame(columns = DF_COLUMNS)
    for file in csv_files:
        data = pd.read_csv(os.path.join(benchmark_csv_files_dir,file),index_col=False)
        average_time = data.mean()
        if 'train' in file and 'half' in file:
            df = insert_series_to_df(df,env,gpu,'train','half',average_time)
        elif 'train' in file and 'float' in file:
            df = insert_series_to_df(df,env,gpu,'train','float',average_time)
        elif 'train' in file and 'double' in file:
            df = insert_series_to_df(df,env,gpu,'train','double',average_time)
        elif 'inference'in file and 'half' in file:
            df = insert_series_to_df(df,env,gpu,'inference','half',average_time)
        elif 'inference'in file and 'float' in file:
            df = insert_series_to_df(df,env,gpu,'inference','float',average_time)
        elif 'inference'in file and 'double' in file:
            df = insert_series_to_df(df,env,gpu,'inference','double',average_time)
        else:
            raise Exception("error file: ",file)
    return df

def plot_models_on_same_gpu(models_time_info,gpu,save_image_dir):
    # 在train、infrence 过程中，三种数据类型下的各个 model 的运行时间。总共6张图。
    for phase in phases:
        for precision in precisions:
            plt_name = "{}_{}_{}.png".format(gpu,phase,precision)
            plt_save_path = os.path.join(save_image_dir,plt_name)
            
            models_time_info.sort_values(by=['models'], inplace=True, ascending=False)
            
            names = models_time_info[(models_time_info.phases == phase) & (models_time_info.precisions == precision)]['models'].tolist()
            times = models_time_info[(models_time_info.phases == phase) & (models_time_info.precisions == precision)]['time'].tolist()
            
            name_index = [i for i in range(len(names))]
            plt.figure(figsize=(20, 10), dpi=100)
            plt.barh(name_index,times,color='b',alpha=0.4)
            plt.yticks(name_index, names,fontsize=14)
            for i in range(len(names)):
                plt.text(times[i],name_index[i]-0.25, times[i] , fontsize=12)
                        
            plt.xlabel('Time(ms)', fontsize=18)
            plt.ylabel('Model', fontsize=16)
            plt_title = '{} {} models with {} precision'.format(gpu,phase,precision) 
            plt.suptitle(plt_title, fontsize=20)
            plt.savefig(plt_save_path)

def plot_image_with_models_benchmark_on_special_gpu_between_envs(gpu,phase,precision,big_data_frame):
    
    df_envs_models_time = big_data_frame[(big_data_frame.gpus==gpu) & (big_data_frame.phases==phase) & (big_data_frame.precisions==precision)]
    df_envs_models_time = df_envs_models_time.sort_values(['envs','models'])
    
    models = []
    envs = list(set(df_envs_models_time['envs'].tolist()))
    
    envs_time_dict = {}
    for index, rows in df_envs_models_time.iterrows():
        # print(index,rows)
        if rows.envs in envs_time_dict:
            envs_time_dict[rows.envs].append(rows.time)
        else:
            envs_time_dict[rows.envs] = [rows.time]
        if rows.models not in models:
            models.append(rows.models)

    plotdata = pd.DataFrame(envs_time_dict,index = models)
    plotdata = plotdata.sort_index(axis = 1)
    plotdata = plotdata.sort_index(axis = 0)
    
    plotdata.plot(figsize=(30,13),kind="bar",rot=-15,alpha=0.4)

    plt.xlabel("Models",fontsize=14)
    plt.ylabel("Time",fontsize=14)

    plt.title('{} {} models with {} precision'.format(gpu,phase,precision),fontsize=16)
    
    benchmark_images_save_dir = ENVS_BENCHMARK_IMAGE_SAVE_DIR + '/' + '{}_benchmark_bewteen_{}'.format(gpu,"_".join(sorted(envs)))
    if not os.path.exists(benchmark_images_save_dir):
        os.makedirs(benchmark_images_save_dir)
    
    plt_image_name = '{} {}_models_with_{}_precision_between_{}'.format(gpu,phase,precision,"_".join(sorted(envs)))
    save_path = os.path.join(benchmark_images_save_dir,plt_image_name)
    plt.savefig(save_path)
    
 
def plot_image_for_compare_model_benchmark_on_multiple_gpus(env,phase,precision,big_data_frame):
    gpu_benchmark_images_save_dir = GPUS_BENCHMARK_IMAGE_SAVE_DIR
    if not os.path.exists(gpu_benchmark_images_save_dir):
        os.makedirs(gpu_benchmark_images_save_dir)
    df_gpus_models_time = big_data_frame[(big_data_frame.envs ==env) & (big_data_frame.phases == phase) & (big_data_frame.precisions == precision)]
    df_gpus_models_time = df_gpus_models_time.sort_values(['gpus','models'])
    
    models = []
    gpus = list(set(df_gpus_models_time['gpus'].tolist()))
    
    gpus_time_dict = {}
    for index, rows in df_gpus_models_time.iterrows():
        if rows.gpus in gpus_time_dict:
            gpus_time_dict[rows.gpus].append(rows.time)
        else:
            gpus_time_dict[rows.gpus] = [rows.time]
        if rows.models not in models:
            models.append(rows.models)

    plotdata = pd.DataFrame(gpus_time_dict,index = models)
    plotdata = plotdata.sort_index(axis = 1)
    plotdata = plotdata.sort_index(axis = 0)
  
    plotdata.plot(figsize=(30,13),kind="bar",rot=-15,alpha=0.4)
    plt.xlabel("Models",fontsize=14)
    plt.ylabel("Time",fontsize=14)

    plt.title('{} models with {} precision on multiples gpus'.format(phase,precision),fontsize=16)
    plt_image_name = '{}_models_with_{}_precision_on_gpus_{}'.format(phase,precision,"_".join(sorted(gpus)))
    save_path = os.path.join(gpu_benchmark_images_save_dir,plt_image_name)
    plt.savefig(save_path)






def run_models_on_special_gpu():
    if len(args.ENVS) != 1:
        raise Exception("env nums must be one, but current nums is : {}".format(len(args.ENVS)),args.ENVS)
    env=args.ENVS[0]
    nvml.nvmlInit()
    gpu_count = nvml.nvmlDeviceGetCount()
    gpu="".join((device_name.replace(" ","_"), '_' , str(gpu_count), '_gpus'))
    
    experiment(env,gpu)

    experiment_result = RESULT_ROOT_DIR
    benchmark_csv_files_dir = os.path.join(experiment_result,env,gpu,'data')
    benchmark_data_frame_gpu_level = build_small_data_frame_for_benchmark(env,gpu,benchmark_csv_files_dir)
    
    save_image_dir = os.path.join(experiment_result,env,gpu,'images')
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    plot_models_on_same_gpu(benchmark_data_frame_gpu_level,gpu,save_image_dir)


def build_gpus_data_frame_benchmark(experiment_result,env,gpus):
    gpus_data_frame = pd.DataFrame(columns = DF_COLUMNS)
    for gpu in gpus:
        benchmark_csv_files_dir = os.path.join(experiment_result,env,gpu,'data')
        small_data_frame = build_small_data_frame_for_benchmark(env,gpu,benchmark_csv_files_dir)
        gpus_data_frame = pd.concat([gpus_data_frame, small_data_frame], axis=0, ignore_index=True)
    return gpus_data_frame

def statistic_benchmark_between_gpus():
    experiment_result = RESULT_ROOT_DIR
    if len(args.ENVS) != 1:
        raise Exception("env nums must be one, but current nums is : {}".format(len(args.ENVS)), args.ENVS)
    env=args.ENVS[0]
    if len(args.GPUS) < 2:
        raise Exception("gpu nums must more one, but current nums is : {}".format(len(args.GPUS)), args.GPUS)
    gpus = args.GPUS
    existed_gpus_dir  = os.path.join(experiment_result,env)
    existed_gpus = get_sub_dir(existed_gpus_dir)
    for gpu in gpus:
        if gpu not in existed_gpus:
            raise Exception("{} not existed !".format(gpu))

    gpus_data_frame = build_gpus_data_frame_benchmark(experiment_result,env,gpus)
    for phase in phases:
        for precision in precisions:
            plot_image_for_compare_model_benchmark_on_multiple_gpus(env,phase,precision,gpus_data_frame)


def validate_args_for_statistic_benchmark_between_envs():
    experiment_result = RESULT_ROOT_DIR
    if len(args.ENVS) < 2:
        raise Exception("env nums must more one, but current nums is : {}".format(len(args.ENVS)), args.ENVS)
    envs = args.ENVS
    if len(args.GPUS) != 1:
        raise Exception("gpu nums must be one, but current nums is : {}".format(len(args.GPUS)), args.GPUS)
    gpu = args.GPUS[0]

    existed_envs = get_sub_dir(experiment_result)
    for env in envs:
        if env not in existed_envs:
            raise Exception("{} not existed !".format(env))
        existed_gpus = get_sub_dir(os.path.join(experiment_result,env))
        if gpu not in existed_gpus:
            raise Exception("{} not existed in {} !".format(gpu, env))
    return envs, gpu

def build_envs_data_frame_benchmark(experiment_result,envs,gpu):
    envs_data_frame = pd.DataFrame(columns = DF_COLUMNS)
    for env in envs:
        benchmark_csv_files_dir = os.path.join(experiment_result,env,gpu,'data')
        small_data_frame = build_small_data_frame_for_benchmark(env,gpu,benchmark_csv_files_dir)
        envs_data_frame = pd.concat([envs_data_frame, small_data_frame], axis=0, ignore_index=True)
    return envs_data_frame    

def statistic_benchmark_between_envs():
    envs, gpu  = validate_args_for_statistic_benchmark_between_envs()
    envs_data_frame = build_envs_data_frame_benchmark(RESULT_ROOT_DIR,envs,gpu)

    for phase in phases:
        for precision in precisions:
            plot_image_with_models_benchmark_on_special_gpu_between_envs(gpu,phase,precision,envs_data_frame)


if __name__ == '__main__':

    if args.opertation  == 'run':
        run_models_on_special_gpu()
    elif args.opertation == 'gpu':
        statistic_benchmark_between_gpus()
    elif args.opertation == 'env':
        statistic_benchmark_between_envs()
    else:
        raise Exception("error opertation !",args.opertation)
