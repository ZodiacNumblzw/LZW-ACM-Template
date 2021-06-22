//FWT                sjf����
namespace FWT
{
	ll inv2;//2��p����Ԫ ����xor FWT��任�õ�
	const ll p=1e9+7;          //������ȡģʱ��ע�����Ԫ�滻��
	ll pow2(ll a,ll b)
	{
		ll res=1;
		while(b)
		{
			if(b&1) res=res*a%p;
			a=a*a%p;
			b>>=1;
		}
		return res;
	}
    //��a�������FWT�任 n����Ϊ2^m��ʽ
    //��f==1��xor_FWT,��f==2��and_FWT,��f==3��or_FWT
    //v=0ʱ������FWT�任��v=1ʱ����任
	void fwt(ll *a,int n,int f,int v)
	{  
		for(int d=1;d<n;d<<=1)
		{
			for(int m=d<<1,i=0;i<n;i+=m)
			{
				for(int j=0;j<d;j++)
				{  
					ll x=a[i+j],y=a[i+j+d];
					if(!v)
					{
						if(f==1) a[i+j]=(x+y)%p,a[i+j+d]=(x-y+p)%p;//xor
						else if(f==2) a[i+j]=(x+y)%p;//and
						else if(f==3) a[i+j+d]=(x+y)%p;//or
					}
					else
					{
						if(f==1) a[i+j]=(x+y)*inv2%p,a[i+j+d]=(x-y+p)%p*inv2%p;//xor
						else if(f==2) a[i+j]=(x-y+p)%p;//and
						else if(f==3) a[i+j+d]=(y-x+p)%p;//or
					}
				}
			}
		}
	}
	
	//�������a 
	void XOR(ll *a,ll *b,int n)
	{
		int len;
		for(len=1;len<=n;len<<=1);
		fwt(a,len,1,0);
		fwt(b,len,1,0);
		for(int i=0;i<len;i++) a[i]=a[i]*b[i]%p;
		inv2=pow2(2,p-2);
		fwt(a,len,1,1);
	}
	void AND(ll *a,ll *b,int n)
	{
		int len;
		for(len=1;len<=n;len<<=1);
		fwt(a,len,2,0);
		fwt(b,len,2,0);
		for(int i=0;i<len;i++) a[i]=a[i]*b[i]%p;
		fwt(a,len,2,1);
	}
	void OR(ll *a,ll *b,int n)
	{
		int len;
		for(len=1;len<=n;len<<=1);
		fwt(a,len,3,0);
		fwt(b,len,3,0);
		for(int i=0;i<len;i++) a[i]=a[i]*b[i]%p;
		fwt(a,len,3,1);
	}
};








//yyb��FWT���� orz
void FWT_or(int *a,int opt)
{
    for(int i=1; i<N; i<<=1)
        for(int p=i<<1,j=0; j<N; j+=p)
            for(int k=0; k<i; ++k)
                if(opt==1)a[i+j+k]=(a[j+k]+a[i+j+k])%MOD;
                else a[i+j+k]=(a[i+j+k]+MOD-a[j+k])%MOD;
}
void FWT_and(int *a,int opt)
{
    for(int i=1; i<N; i<<=1)
        for(int p=i<<1,j=0; j<N; j+=p)
            for(int k=0; k<i; ++k)
                if(opt==1)a[j+k]=(a[j+k]+a[i+j+k])%MOD;
                else a[j+k]=(a[j+k]+MOD-a[i+j+k])%MOD;
}
void FWT_xor(int *a,int opt)
{
    for(int i=1; i<N; i<<=1)
        for(int p=i<<1,j=0; j<N; j+=p)
            for(int k=0; k<i; ++k)
            {
                int X=a[j+k],Y=a[i+j+k];
                a[j+k]=(X+Y)%MOD;
                a[i+j+k]=(X+MOD-Y)%MOD;
                if(opt==-1)a[j+k]=1ll*a[j+k]*inv2%MOD,a[i+j+k]=1ll*a[i+j+k]*inv2%MOD;
            }
}