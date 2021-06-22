/*
	快速读入
*/

//普通读入挂
template<class T>
inline bool scan_d(T &ret)
{
	char c;
	int sgn;
	if(c=getchar(),c==EOF)  return 0;
	while (c != '-' && (c < '0' || c > '9') ) c = getchar();
	    sgn = (c == '-') ? -1 : 1;
	    ret = (c == '-') ? 0 : (c - '0');
 while (c = getchar(), c >= '0' && c <= '9') ret = ret * 10 + (c - '0');
			    ret *= sgn;
			    return 1;
}





//fread读入挂
inline char gc()//用fread把读入加速到极致，具体我也不懂，背板子就好
{
    static char BB[1000000],*S=BB,*T=BB;
    return S==T&&(T=(S=BB)+fread(BB,1,1000000,stdin),S==T)?EOF:*S++;
}
inline int getin()//getchar换成fread的快读
{
    register int x=0;register char ch=gc();
    while(ch<48)ch=gc();
    while(ch>=48)x=x*10+(ch^48),ch=gc();
    return x;
}



//read读入挂
template <typename _tp> inline _tp read(_tp&x){
	char ch=getchar(),sgn=0;x=0;
	while(ch^'-'&&!isdigit(ch))ch=getchar();if(ch=='-')ch=getchar(),sgn=1;
	while(isdigit(ch))x=x*10+ch-'0',ch=getchar();if(sgn)x=-x;return x;
}

int main(){int a,b;b=read(a);}  //read()即可读入a 可再赋值给b

//输入输出挂
template <class T>
inline bool read(T &ret)
{
    char c;
    int sgn;
    T bit=0.1;
    if(c=getchar(),c==EOF) return 0;
    while(c!='-'&&c!='.'&&(c<'0'||c>'9')) c=getchar();
    sgn=(c=='-')?-1:1;
    ret=(c=='-')?0:(c-'0');
    while(c=getchar(),c>='0'&&c<='9') ret=ret*10+(c-'0');
    if(c==' '||c=='\n')
    {
        ret*=sgn;
        return 1;
    }
    while(c=getchar(),c>='0'&&c<='9') ret+=(c-'0')*bit,bit/=10;
    ret*=sgn;
    return 1;
}
inline void out(int x)
{
    if(x>9) out(x/10);
    putchar(x%10+'0');
}