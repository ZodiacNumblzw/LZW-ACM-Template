//AC自动机求长度为n的包含一堆串中任一个串的字符串的个数
//AC自动机+矩阵快速幂模板 HDU2243
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define Accepted 0
#define ull unsigned long long
#define ll unsigned long long
using namespace std;
struct Matrix                       //矩阵结构体
{
    int sizen;
    ull m[105][105];
    void clear()
    {
        sizen=0;
        memset(m,0,sizeof(m));
    }
} M1,M2;

ll quick(ll a,ll b,ll m)
{
    ll ans=1;
    while(b)
    {
        if(b&1)
        {
            ans=ans*a;
        }
        a=a*a;
        b>>=1;
    }
    return ans;
}
namespace Matrix_quick
{
inline Matrix multi(Matrix a,Matrix b)           //矩阵乘法
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(register int i=0; i<=sizen; i++)
        for(register int j=0; j<=sizen; j++)
        {
            for(register int k=0; k<=sizen; k++)
                c.m[i][j]=(c.m[i][j]+a.m[i][k]*b.m[k][j]);
        }
    return c;
}
inline Matrix quickpow(Matrix a,ll b)                   //矩阵二进制快速幂
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(register int i=0; i<=sizen; i++)                            //初始化为单位矩阵
        c.m[i][i]=1;
    while(b)
    {
        if(b&1)
            c=multi(c,a);                                       //快速幂
        a=multi(a,a);
        b>>=1;
    }
    return c;
}
}
struct Aho_Corasick_Automaton
{
#define MAX 105
#define K 26                  //may need change
#define type int
    /***********************/
//basic
    //nxt为构建字典树的数组,fail维护失配时的转移后的结点的下标
    int nxt[MAX][K],fail[MAX];
    int root,tot;        //根节点下标和字典树中结点的数量
//special 结点中维护的信息，和结点同步更新
    type v[MAX];
    /***********************/
    inline int getid(char c)
    {
        return c-'a';     //may need change
    }
    inline void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    //从内存池中新建一个结点
    inline int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        //////
        fail[tot]=0;
        v[tot]=0;
        return tot;
    }
    //向Trie中插入一个字符串
    inline void insert(string str)
    {
        int now = root;
        int len=str.size();
        for (register int i=0; i<len; i++)
        {
            int id = getid(str[i]);
            if(!nxt[now][id])nxt[now][id] = newnode();
            now = nxt[now][id];
            if(i==len-1)
                v[now]=1;
        }
    }
    //BFS建立fail指针
    inline void build()
    {
        //root的fail为自己
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (register int i=0; i<K; i++)          ////
            {
                if(nxt[head][i])
                {
                    Q.push(nxt[head][i]);
                    //此处fail不能指向自己，需要特判
                    /*
                    4 10
                    ATAA
                    TAA
                    AA
                    A
                    */
                    if(nxt[fail[head]][i]!=nxt[head][i])
                        fail[nxt[head][i]]=nxt[fail[head]][i];
                }
                else
                {
                    nxt[head][i]=nxt[fail[head]][i];
                }
                v[nxt[head][i]]|=v[nxt[fail[head]][i]];
            }
        }
    }
    inline void init()
    {
        M1.sizen=M2.sizen=tot+1;
        for(register int i=0; i<=tot; i++)
        {
            for(register int j=0; j<K; j++)
            {
                if(!v[i]&&!v[nxt[i][j]])
                {
                    M1.m[i][nxt[i][j]]++;
                }
                M2.m[i][nxt[i][j]]++;
            }
        }
        for(int i=0; i<=tot+1; i++)
            M1.m[i][tot+1]=1,M2.m[i][tot+1]=1;
         /*for(int i=0;i<=tot+1;i++)
            cout<<v[i]<<endl;
        for(int i=0;i<=tot+1;i++)
        {
            for(int j=0;j<=tot+1;j++)
            {
                cout<<M.m[i][j]<<' ';
            }
            cout<<endl;
        }*/
    }
#undef type
#undef MAX
} ac;

string str;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    register ull m,n;
    while(cin>>m>>n)
    {
        ac.clear();
        M1.clear();
        M2.clear();
        for(register int i=1; i<=m; i++)
            cin>>str,ac.insert(str);
        ac.build();
        ac.init();
        M1=Matrix_quick::quickpow(M1,n+1);
        M2=Matrix_quick::quickpow(M2,n+1);
        int sz=M1.sizen;
        /*for(int i=0;i<=sz;i++)
        {
            for(int j=0;j<=sz;j++)
            {
                cout<<M.m[i][j]<<' ';
            }
            cout<<endl;
        }*/
        ull ans=(M2.m[0][sz]-M1.m[0][sz]);
        cout<<ans<<endl;
    }
    return 0;
}