function lgraph = unet_resnet50_se_BS(inputSize,show)

lgraph = layerGraph();
tempLayers = [
    imageInputLayer(inputSize,"Name","imageinput","Normalization","rescale-zero-one")
    convolution2dLayer([7 7],64,"Name","conv","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm","Epsilon",0.001)
    reluLayer("Name","relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","maxpool","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_1","Epsilon",0.001)
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],64,"Name","conv_2","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_2","Epsilon",0.001)
    reluLayer("Name","relu_2")
    convolution2dLayer([1 1],256,"Name","conv_3","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_3","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_4","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_4","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_5","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_5","Epsilon",0.001)
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],64,"Name","conv_6","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_6","Epsilon",0.001)
    reluLayer("Name","relu_5")
    convolution2dLayer([1 1],256,"Name","conv_7","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_7","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_8","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_8","Epsilon",0.001)
    reluLayer("Name","relu_7")
    convolution2dLayer([3 3],64,"Name","conv_9","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_9","Epsilon",0.001)
    reluLayer("Name","relu_8")
    convolution2dLayer([1 1],256,"Name","conv_10","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_10","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_11","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_11","Epsilon",0.001)
    reluLayer("Name","relu_10")
    convolution2dLayer([3 3],128,"Name","conv_12","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_12","Epsilon",0.001)
    reluLayer("Name","relu_11")
    convolution2dLayer([1 1],512,"Name","conv_13","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_13","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","conv_14","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_14","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_15","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_15","Epsilon",0.001)
    reluLayer("Name","relu_13")
    convolution2dLayer([3 3],128,"Name","conv_16","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_16","Epsilon",0.001)
    reluLayer("Name","relu_14")
    convolution2dLayer([1 1],512,"Name","conv_17","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_17","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_18","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_18","Epsilon",0.001)
    reluLayer("Name","relu_16")
    convolution2dLayer([3 3],128,"Name","conv_19","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_19","Epsilon",0.001)
    reluLayer("Name","relu_17")
    convolution2dLayer([1 1],512,"Name","conv_20","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_20","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5")
    reluLayer("Name","relu_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_21","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_21","Epsilon",0.001)
    reluLayer("Name","relu_19")
    convolution2dLayer([3 3],128,"Name","conv_22","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_22","Epsilon",0.001)
    reluLayer("Name","relu_20")
    convolution2dLayer([1 1],512,"Name","conv_23","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_23","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_6")
    reluLayer("Name","relu_21")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_24","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_24","Epsilon",0.001)
    reluLayer("Name","relu_22")
    convolution2dLayer([3 3],256,"Name","conv_25","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_25","Epsilon",0.001)
    reluLayer("Name","relu_23")
    convolution2dLayer([1 1],1024,"Name","conv_26","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_26","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","conv_27","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_27","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_7")
    reluLayer("Name","relu_24")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_28","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_28","Epsilon",0.001)
    reluLayer("Name","relu_25")
    convolution2dLayer([3 3],256,"Name","conv_29","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_29","Epsilon",0.001)
    reluLayer("Name","relu_26")
    convolution2dLayer([1 1],1024,"Name","conv_30","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_30","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_8")
    reluLayer("Name","relu_27")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_31","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_31","Epsilon",0.001)
    reluLayer("Name","relu_28")
    convolution2dLayer([3 3],256,"Name","conv_32","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_32","Epsilon",0.001)
    reluLayer("Name","relu_29")
    convolution2dLayer([1 1],1024,"Name","conv_33","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_33","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_9")
    reluLayer("Name","relu_30")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_34","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_34","Epsilon",0.001)
    reluLayer("Name","relu_31")
    convolution2dLayer([3 3],256,"Name","conv_35","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_35","Epsilon",0.001)
    reluLayer("Name","relu_32")
    convolution2dLayer([1 1],1024,"Name","conv_36","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_36","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_10")
    reluLayer("Name","relu_33")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_37","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_37","Epsilon",0.001)
    reluLayer("Name","relu_34")
    convolution2dLayer([3 3],256,"Name","conv_38","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_38","Epsilon",0.001)
    reluLayer("Name","relu_35")
    convolution2dLayer([1 1],1024,"Name","conv_39","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_39","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_11")
    reluLayer("Name","relu_36")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_40","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_40","Epsilon",0.001)
    reluLayer("Name","relu_37")
    convolution2dLayer([3 3],256,"Name","conv_41","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_41","Epsilon",0.001)
    reluLayer("Name","relu_38")
    convolution2dLayer([1 1],1024,"Name","conv_42","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_42","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_12")
    reluLayer("Name","relu_39")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","conv_43","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_43","Epsilon",0.001)
    reluLayer("Name","relu_40")
    convolution2dLayer([3 3],512,"Name","conv_44","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_44","Epsilon",0.001)
    reluLayer("Name","relu_41")
    convolution2dLayer([1 1],2048,"Name","conv_45","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_45","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","conv_46","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_46","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_13")
    reluLayer("Name","relu_42")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","conv_47","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_47","Epsilon",0.001)
    reluLayer("Name","relu_43")
    convolution2dLayer([3 3],512,"Name","conv_48","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_48","Epsilon",0.001)
    reluLayer("Name","relu_44")
    convolution2dLayer([1 1],2048,"Name","conv_49","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_49","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_14")
    reluLayer("Name","relu_45")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","conv_50","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_50","Epsilon",0.001)
    reluLayer("Name","relu_46")
    convolution2dLayer([3 3],512,"Name","conv_51","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_51","Epsilon",0.001)
    reluLayer("Name","relu_47")
    convolution2dLayer([1 1],2048,"Name","conv_52","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batchnorm_52","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_15")
    transposedConv2dLayer([2 2],1024,"Name","transposed-conv","BiasLearnRateFactor",0,"Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(4,"Name","fc")
    reluLayer("Name","relu_57")
    fullyConnectedLayer(64,"Name","fc_1")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1")
    fullyConnectedLayer(16,"Name","fc_2")
    reluLayer("Name","relu_58")
    fullyConnectedLayer(256,"Name","fc_3")
    sigmoidLayer("Name","sigmoid_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_2")
    fullyConnectedLayer(32,"Name","fc_4")
    reluLayer("Name","relu_59")
    fullyConnectedLayer(512,"Name","fc_5")
    sigmoidLayer("Name","sigmoid_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_3")
    fullyConnectedLayer(64,"Name","fc_6")
    reluLayer("Name","relu_60")
    fullyConnectedLayer(1024,"Name","fc_7")
    sigmoidLayer("Name","sigmoid_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","crop2d");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat")
    convolution2dLayer([3 3],1024,"Name","conv_53","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_53","Epsilon",0.001)
    reluLayer("Name","relu_48")
    convolution2dLayer([3 3],1024,"Name","conv_54","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_54","Epsilon",0.001)
    reluLayer("Name","relu_49")
    transposedConv2dLayer([2 2],512,"Name","transposed-conv_1","BiasLearnRateFactor",0,"Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","crop2d_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_1")
    convolution2dLayer([3 3],512,"Name","conv_55","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_55","Epsilon",0.001)
    reluLayer("Name","relu_50")
    convolution2dLayer([3 3],512,"Name","conv_56","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_56","Epsilon",0.001)
    reluLayer("Name","relu_51")
    transposedConv2dLayer([2 2],256,"Name","transposed-conv_2","BiasLearnRateFactor",0,"Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","crop2d_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_2")
    convolution2dLayer([3 3],256,"Name","conv_57","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_57","Epsilon",0.001)
    reluLayer("Name","relu_52")
    convolution2dLayer([3 3],256,"Name","conv_58","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_58","Epsilon",0.001)
    reluLayer("Name","relu_53")
    transposedConv2dLayer([2 2],128,"Name","transposed-conv_3","BiasLearnRateFactor",0,"Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","crop2d_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_3")
    convolution2dLayer([3 3],128,"Name","conv_59","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_59","Epsilon",0.001)
    reluLayer("Name","relu_54")
    convolution2dLayer([3 3],64,"Name","conv_60","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batchnorm_60","Epsilon",0.001)
    reluLayer("Name","relu_55")
    convolution2dLayer([3 3],1,"Name","conv_61","BiasLearnRateFactor",0,"Padding","same")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,"relu","maxpool");
lgraph = connectLayers(lgraph,"relu","gapool");
lgraph = connectLayers(lgraph,"relu","multiplication/in2");
lgraph = connectLayers(lgraph,"maxpool","conv_1");
lgraph = connectLayers(lgraph,"maxpool","conv_4");
lgraph = connectLayers(lgraph,"batchnorm_3","addition/in1");
lgraph = connectLayers(lgraph,"batchnorm_4","addition/in2");
lgraph = connectLayers(lgraph,"relu_3","conv_5");
lgraph = connectLayers(lgraph,"relu_3","addition_1/in1");
lgraph = connectLayers(lgraph,"batchnorm_7","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_6","conv_8");
lgraph = connectLayers(lgraph,"relu_6","addition_2/in1");
lgraph = connectLayers(lgraph,"batchnorm_10","addition_2/in2");
lgraph = connectLayers(lgraph,"relu_9","conv_11");
lgraph = connectLayers(lgraph,"relu_9","conv_14");
lgraph = connectLayers(lgraph,"relu_9","gapool_1");
lgraph = connectLayers(lgraph,"relu_9","multiplication_1/in2");
lgraph = connectLayers(lgraph,"batchnorm_13","addition_3/in1");
lgraph = connectLayers(lgraph,"batchnorm_14","addition_3/in2");
lgraph = connectLayers(lgraph,"relu_12","conv_15");
lgraph = connectLayers(lgraph,"relu_12","addition_4/in1");
lgraph = connectLayers(lgraph,"batchnorm_17","addition_4/in2");
lgraph = connectLayers(lgraph,"relu_15","conv_18");
lgraph = connectLayers(lgraph,"relu_15","addition_5/in1");
lgraph = connectLayers(lgraph,"batchnorm_20","addition_5/in2");
lgraph = connectLayers(lgraph,"relu_18","conv_21");
lgraph = connectLayers(lgraph,"relu_18","addition_6/in1");
lgraph = connectLayers(lgraph,"batchnorm_23","addition_6/in2");
lgraph = connectLayers(lgraph,"relu_21","conv_24");
lgraph = connectLayers(lgraph,"relu_21","conv_27");
lgraph = connectLayers(lgraph,"relu_21","gapool_2");
lgraph = connectLayers(lgraph,"relu_21","multiplication_2/in2");
lgraph = connectLayers(lgraph,"batchnorm_26","addition_7/in1");
lgraph = connectLayers(lgraph,"batchnorm_27","addition_7/in2");
lgraph = connectLayers(lgraph,"relu_24","conv_28");
lgraph = connectLayers(lgraph,"relu_24","addition_8/in1");
lgraph = connectLayers(lgraph,"batchnorm_30","addition_8/in2");
lgraph = connectLayers(lgraph,"relu_27","conv_31");
lgraph = connectLayers(lgraph,"relu_27","addition_9/in1");
lgraph = connectLayers(lgraph,"batchnorm_33","addition_9/in2");
lgraph = connectLayers(lgraph,"relu_30","conv_34");
lgraph = connectLayers(lgraph,"relu_30","addition_10/in1");
lgraph = connectLayers(lgraph,"batchnorm_36","addition_10/in2");
lgraph = connectLayers(lgraph,"relu_33","conv_37");
lgraph = connectLayers(lgraph,"relu_33","addition_11/in1");
lgraph = connectLayers(lgraph,"batchnorm_39","addition_11/in2");
lgraph = connectLayers(lgraph,"relu_36","conv_40");
lgraph = connectLayers(lgraph,"relu_36","addition_12/in1");
lgraph = connectLayers(lgraph,"batchnorm_42","addition_12/in2");
lgraph = connectLayers(lgraph,"relu_39","conv_43");
lgraph = connectLayers(lgraph,"relu_39","conv_46");
lgraph = connectLayers(lgraph,"relu_39","gapool_3");
lgraph = connectLayers(lgraph,"relu_39","multiplication_3/in2");
lgraph = connectLayers(lgraph,"batchnorm_45","addition_13/in1");
lgraph = connectLayers(lgraph,"batchnorm_46","addition_13/in2");
lgraph = connectLayers(lgraph,"relu_42","conv_47");
lgraph = connectLayers(lgraph,"relu_42","addition_14/in1");
lgraph = connectLayers(lgraph,"batchnorm_49","addition_14/in2");
lgraph = connectLayers(lgraph,"relu_45","conv_50");
lgraph = connectLayers(lgraph,"relu_45","addition_15/in1");
lgraph = connectLayers(lgraph,"batchnorm_52","addition_15/in2");
lgraph = connectLayers(lgraph,"transposed-conv","crop2d/ref");
lgraph = connectLayers(lgraph,"transposed-conv","concat/in1");
lgraph = connectLayers(lgraph,"sigmoid","multiplication/in1");
lgraph = connectLayers(lgraph,"multiplication","crop2d_3/in");
lgraph = connectLayers(lgraph,"sigmoid_1","multiplication_1/in1");
lgraph = connectLayers(lgraph,"multiplication_1","crop2d_2/in");
lgraph = connectLayers(lgraph,"sigmoid_2","multiplication_2/in1");
lgraph = connectLayers(lgraph,"multiplication_2","crop2d_1/in");
lgraph = connectLayers(lgraph,"sigmoid_3","multiplication_3/in1");
lgraph = connectLayers(lgraph,"multiplication_3","crop2d/in");
lgraph = connectLayers(lgraph,"crop2d","concat/in2");
lgraph = connectLayers(lgraph,"transposed-conv_1","crop2d_1/ref");
lgraph = connectLayers(lgraph,"transposed-conv_1","concat_1/in1");
lgraph = connectLayers(lgraph,"crop2d_1","concat_1/in2");
lgraph = connectLayers(lgraph,"transposed-conv_2","crop2d_2/ref");
lgraph = connectLayers(lgraph,"transposed-conv_2","concat_2/in1");
lgraph = connectLayers(lgraph,"crop2d_2","concat_2/in2");
lgraph = connectLayers(lgraph,"transposed-conv_3","crop2d_3/ref");
lgraph = connectLayers(lgraph,"transposed-conv_3","concat_3/in1");
lgraph = connectLayers(lgraph,"crop2d_3","concat_3/in2");

if(show==1)
    plot(lgraph);
end
end


