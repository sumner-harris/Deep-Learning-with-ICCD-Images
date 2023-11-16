import torch
import torch.nn as nn

class MixedICCDNet(nn.Module):
    def __init__(self,l1=64,l2=32,param_l1=48,param_out=32,c1=16,c2=24,c3=32):
        super(MixedICCDNet, self).__init__()
        # ICCD imaging feature inputs, the full image size is BATCH,C,frames,H,W where it is N,50,40,40
        self.ICCD_features_ = nn.Sequential(
            #Spatial convolution
            nn.Conv3d(1,64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(64),
            #Temportal convolution
            nn.Conv3d(64,64,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(64,128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(128),
            #Temportal convolution
            nn.Conv3d(128,128,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(128,256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            #Temportal convolution
            nn.Conv3d(256,256,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            nn.Flatten(start_dim=1),
            nn.Linear(256*6*5*5,l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(l1,l2),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(l2,3)
        )
        
        self.parameter_features = nn.Sequential(
            nn.Linear(4,param_l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(param_l1,param_out),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(param_out,3)
        )
        
        self.combined_features_ = nn.Sequential(
            nn.Linear(l2+param_out,c1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c1,c2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c2,c3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c3,3),
        )

    def forward(self,x,y):
        x=self.ICCD_features_(x)
        y=self.parameter_features(y)
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)        
        z = torch.cat((x,y),1)
        z = self.combined_features_(z)        
        return z
    
class ICCDNet(nn.Module):
    def __init__(self,l1=64,l2=32):
        super(ICCDNet, self).__init__()
        # ICCD imaging feature inputs, the full image size is BATCH,C,frames,H,W where it is N,50,40,40
        self.ICCD_features_ = nn.Sequential(
            #Spatial convolution
            nn.Conv3d(1,64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(64),
            #Temportal convolution
            nn.Conv3d(64,64,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(64,128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(128),
            #Temportal convolution
            nn.Conv3d(128,128,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(128,256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            #Temportal convolution
            nn.Conv3d(256,256,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            nn.Flatten(start_dim=1),
            nn.Linear(256*6*5*5,l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(l1,l2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(l2,3)
        )

    def forward(self,x1):
        #ICCD features
        x1=self.ICCD_features_(x1)
        return x1
 
    
class MLP(nn.Module):
    def __init__(self,param_l1=48,param_out=32):
        super(MLP, self).__init__()
        
        self.parameter_features = nn.Sequential(
            nn.Linear(4,param_l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(param_l1,param_out),
            nn.LeakyReLU(inplace=True),
            nn.Linear(param_out,3)
        )

    def forward(self,x1):
        x1=self.parameter_features(x1)
        return x1