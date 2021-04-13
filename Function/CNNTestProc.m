function R=CNNTestProc(Data_test_in,W1,W3,W4)
[~,Source_dim,Data_test_count]=size(Data_test_in);
[~,Conv_dim,ConvK_count]=size(W1);
Feat_dim=Source_dim-Conv_dim+1;
V1=zeros(Feat_dim,Feat_dim,ConvK_count);
Data_rel_out=zeros(10,Data_test_count);
for i=1:Data_test_count
    for j=1:ConvK_count
        V1(:,:,j)=filter2(W1(:,:,j),Data_test_in(:,:,i),'valid');
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
      Data_rel_out(:,i)=y4;
end
R=Data_rel_out;