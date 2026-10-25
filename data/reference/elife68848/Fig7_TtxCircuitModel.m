fiid = 13;
figure(fiid);clf;hold on;
tmax=200;N=1500;
t=linspace(0,tmax,N);dt=tmax/N;
% MODEL PARAMS:
wnum = 50; % number of agents to simulate
gam=[1.5 1.5 1];km=15; kf=5; kr=5; g=1; Ath=1.5; Mth=-0;fb=1; kt=-1.5; kv=1;
flen=cell(1,wnum);psum=flen;tindx=flen; Xend=zeros(1,wnum);
for wi=1:wnum
    Phist=2*pi*rand; % randomly choose initial heading angle
    XS=[1 1 0 0]; % initial condition
    X=XS;
    for ti=2:N
        A=X(1); M=X(2); XC=X(3); YC=X(4);
        % calculate and update Is
        delti=round(.5*N/tmax); % N/tmax=1s delay
        lagi=max(1,(ti-delti));
        Is=((XC-XS(lagi,3))*kt)*(M>Mth);

        lai=1;
        i1=max(1,(size(XS,1)-lai));i2=max(1,(size(XS,1)-lai+1));
        Mhist=[XS(i1,2) XS(i2,2)];
        Mflg=(Mhist(1)>=(Mth))*10+(Mhist(2)>=(Mth));
        Mfhist(ti)=Mhist(2)>Mth;
        Mfv(ti)=Mflg;
        switch Mflg
            case 11 % f->f
                Pcur=Phist(end);
            case 00 % r->r
                Pcur=Phist(end);
            case 10 % f->r
                Pcur=pi+Phist(end); % reversal
                if (Pcur)>2*pi;Pcur=Pcur-2*pi;end
            case 01 % r->f
                Pcur=rand*2*pi;
        end
        if Mflg==10; Mfhist(ti)=-.5;end

        lagi=max(1,size(XS,1)-2);
        fdur=0;rdur=0;
        Mflg=(Mhist(1)>=(Mth))*10+(Mhist(2)>=(Mth));
        if Mflg==11
            [L, NUM] = bwlabeln((XS(:,2)>Mth));
            fdur=length(find(L==NUM))*dt;
        elseif Mflg==00
            [L, NUM] = bwlabeln((XS(:,2)<Mth));
            rdur=length(find(L==NUM))*dt;
        end

        % calculate derivatives per unit step
        dA=gam(1)*(Is+.5+2*normrnd(0,1)+fb/(1+exp(-km*(M-Mth)))-A)*dt;
ff = 1; rf = .7;
        dM=gam(2)*(ff*1/(1+exp(-kf*(A-Ath)))-rf*g/(1+exp(kr*(A-Ath)))+0*rdur-0*fdur-M)*dt;
        dXC=(kv*abs(1)*cos(Pcur))*dt; dYC=(kv*abs(1)*sin(Pcur))*dt;

        dX=[dA dM dXC dYC]; X=X+dX; XS(ti,:)=X;
        Phist(ti)=Pcur;
    end
    % identify forward runs and record their lengths and corresponding heading angles for the current
    % rendition/worm
    [L, NUM] = bwlabeln((smooth(XS(:,2),4)>Mth-0));
    fl = regionprops(L,'area');
    flen{wi}=[]; psum{wi}=[];tindx{wi}=[];
    for fi=1:length(fl) % loop through all forward runs for the current trajectory
        flen{wi}(fi)=fl(fi).Area*dt;
        bid=find(L==fi);
        psum{wi}(fi)=mean(Phist(bid));
        tindx{wi}(fi)=mean(cos(Phist(bid)));
    end
    Xend(wi)=XS(end,3);
    figure(fiid);hold all
    subplot(5,1,1:4);hold all;
    z = zeros(size(XS(:,1)'));col =t ;
    plot(XS(:,3)',XS(:,4)','b')
    rid = Mfhist<1; rid2 = XS(:,2)<Mth;
%     plot(XS(rid2,3)',XS(rid2,4)','r.')
    plot(XS(end,3),XS(end,4),'m.','markersize',10)
end
xm=median(Xend);
%     plot([xm xm],[-50 50],'k','linewidth',2)
plot([0 0]',yw*[-1 1]','k:','linewidth',1.5)
plot(0,0,'ko','markerfacecolor','y')
tkl=.035;  %axis equal
set(gca,'xlim',xspan,'ylim',yw*[-1 1],'ycolor','w','xcolor','w','xtick',-60:20:40,...
    'xticklabel',{},'yticklabel',{},'ticklength',[tkl tkl])

subplot(5,1,5);cla;hold all
[hd,hx] = hist(Xend,xspan(1):5:xspan(2));
hp = hd/sum(hd);
bw = .65; fc = 'k';
bar(hx,hp,'barwidth',bw,'facecolor',fc)
yrng=get(gca,'ylim');
set(gca,'xlim',xspan)

plot([0 0],[0 1],'k:','linewidth',1.5)
plot(xspan,[0 0],'k','linewidth',.5)
set(gca,'xticklabel',{},'yticklabel',{},...
    'ticklength',[tkl tkl],'ylim',[0 .35],'tickdir','out')
setfigsiz(fsz)
%%
Phi_tp=[]; Rlen_tp=[]; Tidx_tp=[];
for wi=1:wnum
    Phi_tp=[Phi_tp psum{wi}];
    Rlen_tp=[Rlen_tp flen{wi}];
    Tidx_tp=[Tidx_tp -tindx{wi}];
end
aind=find(Phi_tp>pi); Phi_tp(aind)=Phi_tp(aind)-2*pi;

% Generate runlength histogram sorted by angle
nc=12;phivec=linspace(-pi,pi,nc+1);
RLv_tp=cell(1,nc);RLm=zeros(nc,1);RLe=zeros(nc,1);angvec=RLm;
for pv=1:(nc)
    pind=find(Phi_tp>=phivec(pv)&Phi_tp<phivec(pv+1));
    RLv_tp{pv}=Rlen_tp(pind);
    angvec(pv)=mean([phivec(pv),phivec(pv+1)]);
    RLm(pv,:)=mean(RLv_tp{pv});
    RLe(pv)=std(RLv_tp{pv})/sqrt(length(RLv_tp{pv}));
end

figure(fiid2);clf;hold all;
errorbar(angvec,RLm,RLe,'.-','markersize',12,'color',[.8 .5 0],'linewidth',1.5)
xlim([-pi pi])

figure(fiid2+1);clf;hold all
Tidxm=sum(Tidx_tp.*Rlen_tp)/sum(Rlen_tp);
Tidxe = std(Tidx_tp.*Rlen_tp)/sqrt(length(Rlen_tp));
bar(1,Tidxm,'barwidth',bw,'facecolor',[.8 .5 0])
errorbar(1,Tidxm,Tidxe,'k.','linewidth',1)
