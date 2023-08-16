#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from skimage import io
from netCDF4 import Dataset

'''
Current_dir=os.getcwd()
sys.path.append('/home/yuzhu/Chaorders/PyLib')
import pylib
'''

def Organize_coordination_3D(xseq, yseq, zseq):
    # xseq is a list which conatins the selected x coordinates
    # yseq is a list which conatins the selected y coordinates
    # zseq is a list which conatins the selected z coordinates 
    # return is a 3*n numpy array, three rows corespond to x, y, z coordination
           
    nx=len(xseq)
    ny=len(yseq)
    nz=len(zseq)
    location=np.zeros((3, nx*ny*nz), dtype='i8')
    xseq=np.array(xseq)
    yseq=np.array(yseq)
    zseq=np.array(zseq)
    X=np.repeat(xseq, ny*nz)
    Y1=np.repeat(yseq, nz)
    Y=np.tile(Y1, nx)
    Z=np.tile(zseq, nx*ny)
    location[0, :]=X
    location[1, :]=Y
    location[2, :]=Z
    return location

def Boundary_Extension_3D(Image, Extension):
    Image_size=Image.shape
    X=Image_size[0]
    Y=Image_size[1]
    Z=Image_size[2]
    Extended_Image=np.zeros((X+2*Extension, Y+2*Extension, Z+2*Extension), dtype=Image.dtype)
    Extended_Image[Extension: Extension+X, Extension: Extension+Y, Extension: Extension+Z]=Image
    Left_Boundary=Extended_Image[:, :, Extension+1:2*Extension+1]
    Left_Boundary=np.flip(Left_Boundary, axis=2)
    Extended_Image[:, :, 0:Extension]=Left_Boundary
    Right_Boundary=Extended_Image[:, :, Z-1:Extension+Z-1]
    Right_Boundary=np.flip(Right_Boundary, axis=2)
    Extended_Image[:, :, Extension+Z:Z+2*Extension]=Right_Boundary
    Up_Boundary=Extended_Image[Extension+1:2*Extension+1, :, :]
    Up_Boundary=np.flip(Up_Boundary, axis=0)
    Extended_Image[0:Extension, :, :]=Up_Boundary
    Down_Boundary=Extended_Image[X-1:Extension+X-1, :, :]
    Down_Boundary=np.flip(Down_Boundary, axis=0)
    Extended_Image[Extension+X:X+2*Extension, :, :]=Down_Boundary
    Front_Boundary=Extended_Image[:, Extension+1:2*Extension+1, :]
    Front_Boundary=np.flip(Front_Boundary, axis=1)
    Extended_Image[:, 0:Extension, :]=Front_Boundary
    Back_Boundary=Extended_Image[:, Y-1:Extension+Y-1, :]
    Back_Boundary=np.flip(Back_Boundary, axis=1)
    Extended_Image[:, Extension+Y:Y+2*Extension, :]=Back_Boundary
    return Extended_Image

def open_image(img_path):
    if os.path.isdir(img_path):
        img_files=os.listdir(img_path)
        img_files.sort()
        data=[]
        for files in img_files:
            img=io.imread(img_path+'/'+files)         
            data.append(img)
        data=np.array(data)
        print('the shape of this image is, ', data.shape)
    else:
        if os.path.splitext(img_path)[-1]=='.nc':
            nc=Dataset(img_path, 'r')
            try:
                data=nc.variables['segmented']
            except:
                data=nc.variables['tomo']
            data=np.array(data)
        else:
            data=io.imread(img_path)
            #data=np.array(data)
        print('the shape of this image is, ', data.shape)
        
    return data

def Save_as_tiff(path, data, filename):
    #path is the folder path where the image will store, e.g., '/home/yzw/Data'
    #data is the 2D or 3D matrix
    #filename is the name of the tiff file, e.g., 'Indiana'
    dimension=data.shape
    if len(dimension)==2:
        if filename=='':
            io.imsave(path+'/'+'new_image.tiff', data)
        else:
            io.imsave(path+'/'+filename+'.tiff', data)
    if len(dimension)==3:
        z=dimension[0]
        length=len(str(z))
        for i in range(z):
            k=str(i)
            k1=k.zfill(length+1)
            io.imsave(path+'/'+filename+'_'+k1+'.tiff', data[i, :, :])

def Extract_3D_statistics_8_cell_centre(seg, num_threshold): 
    #提取体心处的多点统计
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率
    X, Y, Z=seg.shape
    coordi=np.zeros((X, Y, Z), dtype='u1')
    coordi[:X-2, :Y-2, :Z-2]=1
    idx=np.flatnonzero(coordi>0)
    sub=np.unravel_index(idx, (X, Y, Z)) 
    c0=np.zeros((256, ), dtype='u4')
    c1=np.zeros((256, ), dtype='u4')    
    for i in list(range(len(list(sub[0])))):
        xc=sub[0][i]
        yc=sub[1][i]
        zc=sub[2][i]  
        pc=[seg[xc][yc][zc], seg[xc+2][yc][zc], seg[xc+2][yc+2][zc], seg[xc][yc+2][zc], seg[xc][yc][zc+2], seg[xc+2][yc][zc+2], seg[xc+2][yc+2][zc+2], seg[xc][yc+2][zc+2]] 
        pc=int(''.join(map(str, pc)), 2)
        if seg[xc+1][yc+1][zc+1]==0:
            c0[pc]=c0[pc]+1
        else:
            c1[pc]=c1[pc]+1
    statistic=c0/(c0+c1+0.001)
    cn=c0+c1
    idx=np.flatnonzero(cn<num_threshold)
    statistic.flat[idx]=-0.1
    return statistic
   
def Extract_3D_statistics_8_surface_centre_x(seg, num_threshold): 
    #提取顺x方向（垂直yz面）的面心处的多点统计
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率    
    X, Y, Z=seg.shape
    coordi=np.zeros((X, Y, Z), dtype='u1')
    coordi[1:X-1, 1:Y-1, 1:Z-1]=1
    idx=np.flatnonzero(coordi>0)
    sub=np.unravel_index(idx, (X, Y, Z)) 
    c0=np.zeros((256, ), dtype='u4')
    c1=np.zeros((256, ), dtype='u4')    
    for i in list(range(len(list(sub[0])))):
        xc=sub[0][i]
        yc=sub[1][i]
        zc=sub[2][i]  
        pc=[seg[xc][yc-1][zc-1], seg[xc][yc-1][zc+1], seg[xc][yc+1][zc-1], seg[xc][yc+1][zc+1], seg[xc-1][yc][zc], seg[xc+1][yc][zc]] 
        pc=int(''.join(map(str, pc)), 2)
        if seg[xc][yc][zc]==0:
            c0[pc]=c0[pc]+1
        else:
            c1[pc]=c1[pc]+1
    statistic=c0/(c0+c1+0.001)
    cn=c0+c1
    idx=np.flatnonzero(cn<num_threshold)
    statistic.flat[idx]=-0.1
    return statistic 
    
def Extract_3D_statistics_8_surface_centre_y(seg, num_threshold): 
    #提取顺y方向（垂直xz面）的面心处的多点统计
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率    
    seg1=seg.transpose((1,0,2))
    statistic=Extract_3D_statistics_8_surface_centre_x(seg1, num_threshold)
    return statistic
    
def Extract_3D_statistics_8_surface_centre_z(seg, num_threshold): 
    #提取顺z方向（垂直xy面）的面心处的多点统计
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率    
    seg1=seg.transpose((2,1,0))
    statistic=Extract_3D_statistics_8_surface_centre_x(seg1, num_threshold)
    return statistic        
    
def Extract_3D_statistics_8_edge_centre_x(seg, num_threshold): 
    #提取顺x方向（垂直yz面）的边界中心处的多点统计
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率    
    X, Y, Z=seg.shape
    coordi=np.zeros((X, Y, Z), dtype='u1')
    coordi[1:X-1, 1:Y-1, 1:Z-1]=1
    idx=np.flatnonzero(coordi>0)
    sub=np.unravel_index(idx, (X, Y, Z)) 
    c0=np.zeros((256, ), dtype='u4')
    c1=np.zeros((256, ), dtype='u4')    
    for i in list(range(len(list(sub[0])))):
        xc=sub[0][i]
        yc=sub[1][i]
        zc=sub[2][i]  
        pc=[seg[xc-1][yc][zc], seg[xc+1][yc][zc], seg[xc][yc-1][zc], seg[xc][yc+1][zc], seg[xc][yc][zc-1], seg[xc][yc][zc+1]] 
        pc=int(''.join(map(str, pc)), 2)
        if seg[xc][yc][zc]==0:
            c0[pc]=c0[pc]+1
        else:
            c1[pc]=c1[pc]+1
    statistic=c0/(c0+c1+0.001)
    cn=c0+c1
    idx=np.flatnonzero(cn<num_threshold)
    statistic.flat[idx]=-0.1
    return statistic  
    
def Extract_3D_statistics_8_edge_centre_y(seg, num_threshold): 
    #提取顺y方向（垂直xz面）的边界中心处的多点统计
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率           
    seg1=seg.transpose((1,0,2))
    statistic=Extract_3D_statistics_8_edge_centre_x(seg1, num_threshold)
    return statistic
           
def Extract_3D_statistics_8_edge_centre_z(seg, num_threshold): 
    #提取顺y方向（垂直xz面）的边界中心处的多点统计  
    #seg是一个二值图像（其中一相必须是0）
    #num_threshold定义了有效统计样本数，例如num_threshold=10表示若某一Pattern找到的样本数小于10个，则认为该样本不具备统计意义，其对应MPS概率将被赋值为-0.1
    #返回值为每种pattarn中心点为0的概率         
    seg1=seg.transpose((2,1,0))
    statistic=Extract_3D_statistics_8_edge_centre_x(seg1, num_threshold)
    return statistic
    
def Multiscale_3D_statistics(seg, scale, num_threshold):
    p=np.array(list(range(256)))
    p1=np.ones((256,), dtype='u1')
    statistic=np.zeros((scale, 256), dtype='f4')    
    plt.figure
    for i in list(range(scale)):
        step=2**i
        seg=seg[::step, ::step, ::step]
        #sc=Extract_3D_statistics_8_edge_centre_x(seg, num_threshold)
        #sc=Extract_3D_statistics_8_cell_centre(seg, num_threshold)
        sc=Extract_3D_statistics_8_edge_centre_x(seg, num_threshold)
        statistic[i, :]=sc
        idx=sc>=0 
        p1=p1*(idx.astype('u1')) 
        print('The effective statitisc rate of' + str(step)+ ' scale is: ', np.sum(idx.astype('u1'))/256.0)
    #------------------------------------------------计算相似度-------------------------------------------------------
    Distance_cos=np.ones((scale, scale), dtype='f4')
    Distance_eclidean=np.ones((scale, scale), dtype='f4')    
    for i in list(range(scale)):
        for j in list(range(scale)):
            x_seq=statistic[i, :]
            x_idx=x_seq>=0
            x_idx=x_idx.astype('u1')
            y_seq=statistic[j, :]   
            y_idx=y_seq>=0
            y_idx=y_idx.astype('u1')
            inter_idx=x_idx*y_idx
            inter_idx=np.nonzero(inter_idx)  
            x_seq=x_seq[inter_idx] 
            y_seq=y_seq[inter_idx]                  
            cosdis=np.dot(x_seq, y_seq)/np.linalg.norm(x_seq)/np.linalg.norm(y_seq)
            eclidean=np.sqrt(np.sum(np.square(x_seq-y_seq)))
            Distance_cos[i][j]=cosdis
            Distance_eclidean[i][j]=eclidean
    print('cos_distance is ', Distance_cos)
    print('eclidean_distance is ', Distance_eclidean) 
       
    color_map=['k','r','b','g','c','m','k','r'] 
    #================================================画MPS柱状图======================================================           
    bar_width=0.3      
    for i in list(range(scale)):           
        plt.bar(p+i*bar_width, height=statistic[i,:], width=bar_width, color=color_map[i], label='scale=='+str((i+1)*2)) 
    plt.xlabel('Patterns') 
    plt.ylabel('Probability')     
    plt.legend() # 显示图例
    plt.show()  
    #=================================================================================================================
            
    idx=np.flatnonzero(p1>0)     
    statistic1=statistic[:, idx]
    
    #========================================画统计量大于num_threshold的MPS折线图======================================   
    style=['solid', '-','--','-.', ':', 'solid', '-','--','-.', ':']   
    for i in list(range(scale)):    
        plt.plot(np.array(list(range(statistic1.shape[1]))), statistic1[i, :], label='scale=='+str((i+1)*2),color=color_map[i], linewidth=2, linestyle=style[i])
    plt.xlabel('Patterns') 
    plt.ylabel('Probability')          
    plt.legend() # 显示图例    
    plt.show()
    #=================================================================================================================

    bins_num=11 
    bin_hist=np.zeros((scale, bins_num), dtype='u4')   
    for i in list(range(scale)): 
        n, bins, patches = plt.hist(x=statistic1[i, :], bins=bins_num, color=color_map[i],
                                    alpha=0.8, rwidth=0.3) #alpha 是颜色深度 rwidth 条形宽度，bins条形箱的数目  
        bin_hist[i,:]=n
    plt.show()    
    #==============================================画MPS概率统计柱状图=================================================           
    bar_width=0.02 
    pt=np.linspace(0, 1, 11) 
    print('pt is: ', pt)    
    for i in list(range(scale)):
        print('bins is: ', bin_hist[i,:])           
        plt.bar(pt+i*bar_width, height=bin_hist[i,:], width=bar_width, color=color_map[i], label='scale=='+str((i+1)*2)) 
    plt.xlabel('MPS') 
    plt.ylabel('Probability')                
    plt.legend() # 显示图例
    plt.show()  
    #==================================================================================================================                                       
    return statistic
    
def Recon_cell_centre(seg, statistics):
    X, Y, Z=seg.shape
    '''
    coordi=np.zeros((X, Y, Z), dtype='u1')
    coordi[:X-2, :Y-2, :Z-2]=1
    idx=np.flatnonzero(coordi>0)
    sub=np.unravel_index(idx, (X, Y, Z))
    '''
    xseq=np.array(range(X)) 
    #xseq=xseq+1
    xseq=xseq[::2]
    xseq=xseq[:-1]
    
    yseq=np.array(range(Y)) 
    #yseq=yseq+1
    yseq=yseq[::2]
    yseq=yseq[:-1] 
    
    zseq=np.array(range(Z)) 
    #zseq=zseq+1
    zseq=zseq[::2]
    zseq=zseq[:-1]       
    
    sub=Organize_coordination_3D(xseq, yseq, zseq)
    print('shape of sub is: ', sub.shape) 
    
    for i in list(range(sub.shape[1])):
        xc=sub[0][i]
        yc=sub[1][i]
        zc=sub[2][i]  
        pc=[seg[xc][yc][zc], seg[xc+2][yc][zc], seg[xc+2][yc+2][zc], seg[xc][yc+2][zc], seg[xc][yc][zc+2], seg[xc+2][yc][zc+2], seg[xc+2][yc+2][zc+2], seg[xc][yc+2][zc+2]] 
        pc=int(''.join(map(str, pc)), 2)
        t=np.random.uniform()
        if t<=statistics[pc]:
            seg[xc+1][yc+1][zc+1]=0
        else:
            seg[xc+1][yc+1][zc+1]=1
    return seg 
            
def Recon_surface_centre_x(seg, statistics):
    X, Y, Z=seg.shape
    '''
    coordi=np.zeros((X, Y, Z), dtype='u1')
    coordi[1:X-1, 1:Y-1, 1:Z-1]=1
    idx=np.flatnonzero(coordi>0)
    sub=np.unravel_index(idx, (X, Y, Z))  
    '''
    
    xseq=np.array(range(X)) 
    xseq=xseq+2
    xseq=xseq[::2]
    xseq=xseq[:-2]
    
    yseq=np.array(range(Y)) 
    yseq=yseq+1
    yseq=yseq[::2]
    yseq=yseq[:-1] 
    
    zseq=np.array(range(Z)) 
    zseq=zseq+1
    zseq=zseq[::2]
    zseq=zseq[:-1]       
    
    sub=Organize_coordination_3D(xseq, yseq, zseq)     
    print('shape of sub is: ', sub.shape)       
    for i in list(range(sub.shape[1])):
        xc=sub[0][i]
        yc=sub[1][i]
        zc=sub[2][i]  
        pc=[seg[xc][yc-1][zc-1], seg[xc][yc-1][zc+1], seg[xc][yc+1][zc-1], seg[xc][yc+1][zc+1], seg[xc-1][yc][zc], seg[xc+1][yc][zc]] 
        pc=int(''.join(map(str, pc)), 2)
        t=np.random.uniform()
        if t<=statistics[pc]:
            seg[xc][yc][zc]=0
        else:
            seg[xc][yc][zc]=1
    return seg 
    
def Recon_edge_centre_x(seg, statistics):
    X, Y, Z=seg.shape
    '''
    coordi=np.zeros((X, Y, Z), dtype='u1')
    coordi[1:X-1, 1:Y-1, 1:Z-1]=1
    idx=np.flatnonzero(coordi>0)
    sub=np.unravel_index(idx, (X, Y, Z)) 
    pattern=list(range(256))
    c0=np.zeros((256, ), dtype='u4')
    c1=np.zeros((256, ), dtype='u4')  
    '''
    
    xseq=np.array(range(X)) 
    xseq=xseq+1
    xseq=xseq[::2]
    xseq=xseq[:-1]
    
    yseq=np.array(range(Y)) 
    yseq=yseq+2
    yseq=yseq[::2]
    yseq=yseq[:-2] 
    
    zseq=np.array(range(Z)) 
    zseq=zseq+2
    zseq=zseq[::2]
    zseq=zseq[:-2]       
    
    sub=Organize_coordination_3D(xseq, yseq, zseq)               
    print('shape of sub is: ', sub.shape)       
    for i in list(range(sub.shape[1])):
        xc=sub[0][i]
        yc=sub[1][i]
        zc=sub[2][i]  
        pc=[seg[xc-1][yc][zc], seg[xc+1][yc][zc], seg[xc][yc-1][zc], seg[xc][yc+1][zc], seg[xc][yc][zc-1], seg[xc][yc][zc+1]] 
        pc=int(''.join(map(str, pc)), 2)
        t=np.random.uniform()        
        if t<=statistics[pc]:
            seg[xc][yc][zc]=0
        else:
            seg[xc][yc][zc]=1
    return seg 
            
            
def Reconstruction(seg, recon_scale, MPS_existed, save_path):
    if MPS_existed==None:
        sc_cell_centre=Extract_3D_statistics_8_cell_centre(seg, 10)
        sc_surface_centre_x=Extract_3D_statistics_8_surface_centre_x(seg, 10)        
        sc_surface_centre_y=Extract_3D_statistics_8_surface_centre_y(seg, 10)        
        sc_surface_centre_z=Extract_3D_statistics_8_surface_centre_z(seg, 10) 
        sc_edge_centre_x=Extract_3D_statistics_8_edge_centre_x(seg, 10)               
        sc_edge_centre_y=Extract_3D_statistics_8_edge_centre_y(seg, 10) 
        sc_edge_centre_z=Extract_3D_statistics_8_edge_centre_z(seg, 10)
    else:
        pass #load existed MPS
        
    for i in range(recon_scale):
        eseg=Boundary_Extension_3D(seg, 1)
        X, Y, Z=eseg.shape
        seg1=(-1)*np.ones((2*X-1, 2*Y-1, 2*Z-1), dtype='u1')
        seg1[::2, ::2, ::2]=eseg
        eseg1=seg1
        #====================================重构体心===========================
        eseg1=Recon_cell_centre(eseg1, sc_cell_centre)   
        '''
        plt.imshow(eseg1[1,:,:])  
        plt.show()
        plt.imshow(eseg1[:,3,:])  
        plt.show() 
        plt.imshow(eseg1[:,:,5])  
        plt.show() 
        plt.imshow(eseg1[2,:,:])  
        plt.show()  
        '''                                    
        #====================================重构面心x==========================        
        eseg1=Recon_surface_centre_x(eseg1, sc_surface_centre_x)
        '''
        plt.imshow(eseg1[2,:,:])  
        plt.show()   
        '''
        #====================================重构面心y==========================
        eseg1=eseg1.transpose((1,0,2))        
        eseg1=Recon_surface_centre_x(eseg1, sc_surface_centre_y)        
        eseg1=eseg1.transpose((1,0,2))
        #====================================重构面心z==========================
        eseg1=eseg1.transpose((2,1,0))        
        eseg1=Recon_surface_centre_x(eseg1, sc_surface_centre_z)        
        eseg1=eseg1.transpose((2,1,0)) 
        #====================================重构边界中心x======================        
        eseg1=Recon_edge_centre_x(eseg1, sc_edge_centre_x)
        #====================================重构边界中心y======================
        eseg1=eseg1.transpose((1,0,2))        
        eseg1=Recon_edge_centre_x(eseg1, sc_edge_centre_y)      
        eseg1=eseg1.transpose((1,0,2))        
        #====================================重构边界中心z======================
        eseg1=eseg1.transpose((2,1,0))        
        eseg1=Recon_edge_centre_x(eseg1, sc_edge_centre_z)      
        eseg1=eseg1.transpose((2,1,0)) 
        seg=eseg1[2:-2, 2:-2,2:-2] 
        os.mkdir(save_path+'/recon_'+str(2**i))
        Save_as_tiff(save_path+'/recon_'+str(2**i), seg*255, 'MSSR')
    return seg              
                                 
         
if __name__=='__main__':
        
    seg_path='/home/yuzhu/Documents/papers/new_papers/new_papers/Self-Similarity-MPS/code/FS_HR_PNG' #e.g., '/home/Porous_media'
    save_recon_path='/home/yuzhu/Documents/papers/new_papers/new_papers/Self-Similarity-MPS/code/Recon' #e.g., '/home/Recon'    
    seg=open_image(seg_path) 
    seg=seg>0
    seg=seg.astype('u1')
    reconseg=Reconstruction(seg[::4,::4,::4], 2, None, save_recon_path)

    










