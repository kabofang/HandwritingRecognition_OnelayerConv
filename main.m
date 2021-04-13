addpath(genpath('.\Data'))
addpath(genpath('.\Function'))
Start_time=fix(clock);
fprintf('%.2d:%.2d:%.2d\n',Start_time(4),Start_time(5),Start_time(6));
load('MNISTData.mat');
EPOCH=1;
In_dim=28;
ConvK_dim=9;
ConvK_count=20;
Conved_dim=In_dim-ConvK_dim+1;
Pooling_idx=2;
Pooled_dim=Conved_dim/Pooling_idx;
Hide_node=100;
W1=randn(ConvK_dim,ConvK_dim,ConvK_count);
W3=(2*rand(Hide_node,Pooled_dim*Pooled_dim*ConvK_count)-1)/ ...
                    ((Pooled_dim*Pooled_dim*ConvK_count)/Hide_node); 
W4=(2*rand(10,Hide_node)-1)/(Hide_node/10);
for i=1:EPOCH
    [W1,W3,W4]=CNNTrainProc(Data_train_in,Data_train_out,W1,W3,W4);
end
Data_rel_out=CNNTestProc(Data_test_in,W1,W3,W4);
[~,Data_test_count]=size(Data_test_out);
[~,Data_test_out_01]=max(Data_test_out);
[~,Data_rel_out_01]=max(Data_rel_out);
Correct_count=sum(Data_rel_out_01==Data_test_out_01);
fprintf('Accuracy is %f\n',Correct_count/Data_test_count);
End_time=fix(clock);
fprintf('%.2d:%.2d:%.2d\n',End_time(4),End_time(5),End_time(6));