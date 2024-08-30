function nndata=imgimportapp(datapath,pixelstart,pixelend)

path=strcat(datapath,'\');
filegroupdir=dir(path);%
filenum=size(filegroupdir);
frame=filenum(1)-2;
w = waitbar(0,'0%','Name','正在处理...');

temp=zeros(240,321,frame);
for i=1:frame
     temp(:,:,i)=csvread(strcat(path,filegroupdir(i+2).name));
     waitbar(i/frame,w,sprintf('%d / %d',i,frame));
end

temp(:,321,:)=[];

if isempty(pixelstart)
    nndata=temp;
else
    nndata=temp(pixelstart(1):pixelstart(2),pixelend(1):pixelend(2),:);
end
delete(w);
end
