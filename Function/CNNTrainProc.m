function [R1,R2,R3]=CNNTrainProc(Data_train_in,Data_train_out,W1,W3,W4)
alpha=0.0875;
[~,Source_dim,Data_train_count]=size(Data_train_in);
[~,Conv_dim,ConvK_count]=size(W1);
Feat_dim=Source_dim-Conv_dim+1;
Pooling_idx=2;
V1=zeros(Feat_dim,Feat_dim,ConvK_count);
for i=1:Data_train_count
    for j=1:ConvK_count
        V1(:,:,j)=filter2(W1(:,:,j),Data_train_in(:,:,i),'valid');
    end
     Y1=max(0,V1);
     Y2=(Y1(1:2:end,1:2:end,:) ...
          +Y1(1:2:end,2:2:end,:) ...
          +Y1(2:2:end,1:2:end,:) ...
          +Y1(2:2:end,2:2:end,:))/4;
      y2=reshape(Y2,[],1); 
      v3=W3*y2;
      y3=max(0,v3);
      v4=W4*y3;
      y4=SoftMax(v4);
      e4=Data_train_out(:,i)-y4;
      delta4=e4;
      e3=W4'*delta4;
      delta3=(y3>0).*e3;
      e2=W3'*delta3;
      E2=reshape(e2,size(Y2));
      E1=zeros(size(Y1));
      Etemp=E2/(Pooling_idx*Pooling_idx);
      E1(1:2:end,1:2:end,:)=Etemp;
      E1(1:2:end,2:2:end,:)=Etemp;
      E1(2:2:end,1:2:end,:)=Etemp;
      E1(2:2:end,2:2:end,:)=Etemp;
      DELTA1=(V1>0).*E1;
      dW4=alpha*delta4*y3';
      dW3=alpha*delta3*y2';
      dW1=zeros(size(W1));
    for j=1:ConvK_count
      dW1(:,:,j)=alpha*filter2(DELTA1(:,:,j),Data_train_in(:,:,i),'valid');
    end
      W4=W4+dW4;
      W3=W3+dW3;
      W1=W1+dW1;    
end
R1=W1;R2=W3;R3=W4;