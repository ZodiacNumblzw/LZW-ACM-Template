//AC�Զ����󳤶�Ϊn�İ���һ�Ѵ�����һ�������ַ����ĸ���
//AC�Զ���+���������ģ�� HDU2243
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define Accepted 0
#define ull unsigned long long
#define ll unsigned long long
using namespace std;
struct Matrix                       //����ṹ��
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
inline Matrix multi(Matrix a,Matrix b)           //����˷�
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
inline Matrix quickpow(Matrix a,ll b)                   //��������ƿ�����
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(register int i=0; i<=sizen; i++)                            //��ʼ��Ϊ��λ����
        c.m[i][i]=1;
    while(b)
    {
        if(b&1)
            c=multi(c,a);                                       //������
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
    //nxtΪ�����ֵ���������,failά��ʧ��ʱ��ת�ƺ�Ľ����±�
    int nxt[MAX][K],fail[MAX];
    int root,tot;        //���ڵ��±���ֵ����н�������
//special �����ά������Ϣ���ͽ��ͬ������
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
    //���ڴ�����½�һ�����
    inline int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        //////
        fail[tot]=0;
        v[tot]=0;
        return tot;
    }
    //��Trie�в���һ���ַ���
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
    //BFS����failָ��
    inline void build()
    {
        //root��failΪ�Լ�
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
                    //�˴�fail����ָ���Լ�����Ҫ����
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