function depthdata=binreader(datapath,fream,xpixel,ypixel)

path=strcat(datapath,'\');
filegroupdir=dir(path);%
filenum=size(filegroupdir);
dirnum=filenum(1)-2;
w = waitbar(0,'0%','Name','正在处理...');

depthdata=zeros(240,320,fream*dirnum);
for i=1:dirnum
    fid=fopen(strcat(path,filegroupdir(i+2).name),'rb');
    ldata=fread(fid,'float');
    for j=1:fream
        depthdata(:,:,(i-1)*fream+j)=reshape(ldata(((j-1)*76800+1):j*76800),[240 320]);
    end
    waitbar(i/dirnum,w,sprintf('%d / %d',i,dirnum));
    fclose(fid);
end
delete(w);
depthdata=depthdata(xpixel(1):xpixel(2),ypixel(1):ypixel(2),:);

% xpixel=[35 214];
% ypixel=[63 242];