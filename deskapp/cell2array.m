train_traindata=zeros(144,144,31790);
train_labeldata=zeros(144,144,31790);

var_traindata=zeros(144,144,5160);
var_labeldata=zeros(144,144,5160);

for i=1:31790
    train_traindata(:,:,i)=traindataArray{i,1};
    train_labeldata(:,:,i)=traindataArray{i,2};
end
for i=1:5160
    var_traindata(:,:,i)=vardataArray{i,1};
    var_labeldata(:,:,i)=vardataArray{i,2};
end
