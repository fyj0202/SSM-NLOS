%%%% 输入图像尺寸 %%%% ↓
inputsize=[224 224 1];

%%%% 更换网络 %%%% ↓
unet=unet_resnet50_se_BS_dropout(inputsize,0);

%%%% 无需修改训练参数 %%%% % ↓
options = trainingOptions('adam', ... %优化器
'MiniBatchSize',64, ...%128
'Shuffle','every-epoch', ... %每一轮打乱参数训练
'ValidationData',varSet, ...
'MaxEpochs',150, ...
'Plots','training-progress', ... %绘制训练过程
'InitialLearnRate',0.0005, ... %初始学习率确定为0.0005
'LearnRateSchedule','piecewise', ... 
'LearnRateDropFactor',0.90, ...
'LearnRateDropPeriod', 15, ...%即几个epoch下降一次学习率 100就10 200就20
'L2Regularization',0.002, ...%0.002 0.00005
'Verbose',true, ...
'VerboseFrequency', 195, ... 
'ValidationFrequency',195, ...
'ExecutionEnvironment','gpu');
%%%% netinfo:训练过程的详细消息 %%%% ↓
[net,netinfo]=trainNetwork(trainSet,'labelimg',unet,options);