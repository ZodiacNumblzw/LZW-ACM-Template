//SG函数模板        Hdu1536 凡是可以化简为Impartial Combinatorial Games的抽象模型（有向图移动棋子）
//都可以尝试利用SG函数 复杂度目前还不太清楚，板子是抄的
//如果堆数或每堆的棋子数太大，可以考虑打标猜规律找循环节   如Codeforces1194D
#include<bits/stdc++.h>
#include<ext/rope>
using namespace std;
using namespace __gnu_cxx;
#define ll long long
#define ld long double
#define ull unsigned long long
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-6;
const int maxn=1e5+10;
const int INF=0x3f3f3f3f;
const double e=2.718281828459045;
#define Accepted 0
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}

//注意 S数组要按从小到大排序 SG函数要初始化为-1 对于每个集合只需初始化1边
//不需要每求一个数的SG就初始化一边
int SG[10100],n,m,s[102],k;//k是集合s的大小 S[i]是定义的特殊取法规则的数组
int dfs(int x)//求SG[x]模板
{
    if(SG[x]!=-1) return SG[x];
    bool vis[110];
    memset(vis,0,sizeof(vis));

    for(int i=0; i<k; i++)
    {
        if(x>=s[i])
        {
            dfs(x-s[i]);
            vis[SG[x-s[i]]]=1;
        }
    }
    int e;
    for(int i=0;; i++)
        if(!vis[i])
        {
            e=i;
            break;
        }
    return SG[x]=e;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    while(cin>>k&&k)
    {
        memset(SG,-1,sizeof(SG));
        for(int i=0; i<k; i++)
            cin>>s[i];
        cin>>m;
        for(int i=0; i<m; i++)
        {
            int sum=0;
            cin>>n;
            for(int i=0,a; i<n; i++)
            {
                cin>>a;
                sum^=dfs(a);
            }
            // printf("SG[%d]=%d\n",num,SG[num]);
            if(sum==0) putchar('L');
            else putchar('W');
        }
        putchar('\n');
    }
    return Accepted;
}

//还是SG函数
//神仙网友的板子也挺不错的，贴在这里    
//https://www.cnblogs.com/dyllove98/p/3194312.html

#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;
//注意 S数组要按从小到大排序 SG函数要初始化为-1 对于每个集合只需初始化1边
//不需要每求一个数的SG就初始化一边
int SG[10100],n,m,s[102],k;//k是集合s的大小 S[i]是定义的特殊取法规则的数组
int dfs(int x)//求SG[x]模板
{
    if(SG[x]!=-1) return SG[x];
    bool vis[110];
    memset(vis,0,sizeof(vis));

    for(int i=0;i<k;i++)
    {
        if(x>=s[i])
        {
           dfs(x-s[i]);
           vis[SG[x-s[i]]]=1;
         }
    }
    int e;
    for(int i=0;;i++)
      if(!vis[i])
      {
        e=i;
        break;
      }
    return SG[x]=e;
}
int main()
{
    int cas,i;
    while(scanf("%d",&k)!=EOF)
    {
        if(!k) break;
        memset(SG,-1,sizeof(SG));
        for(i=0;i<k;i++) scanf("%d",&s[i]);
        sort(s,s+k);
        scanf("%d",&cas);
        while(cas--)
        {
            int t,sum=0;
            scanf("%d",&t);
            while(t--)
            {
                int num;
                scanf("%d",&num);
                sum^=dfs(num);
               // printf("SG[%d]=%d\n",num,SG[num]);
            }
            if(sum==0) printf("L");
            else printf("W");
        }
        printf("\n");
    }
    return 0;
}


//下面是对SG打表的做法
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
const int K=101;
const int H=10001;//H是我们要打表打到的最大值
int k,m,l,h,s[K],sg[H],mex[K];///k是集合元素的个数 s[]是集合  mex大小大约和集合大小差不多
///注意s的排序
void sprague_grundy()
{
    int i,j;
    sg[0]=0;
    for (i=1; i<H; i++)
    {
        memset(mex,0,sizeof(mex));
        j=1;
        while (j<=k && i>=s[j])
        {
            mex[sg[i-s[j]]]=1;
            j++;
        }
        j=0;
        while (mex[j]) j++;
        sg[i]=j;
    }
}

int main()
{
    int tmp,i,j;
    scanf("%d",&k);
    while (k!=0)
    {
        for (i=1; i<=k; i++)
            scanf("%d",&s[i]);
        sort(s+1,s+k+1);            //这个不能少
        sprague_grundy();
        scanf("%d",&m);
        for (i=0; i<m; i++)
        {
            scanf("%d",&l);
            tmp=0;
            for (j=0; j<l; j++)
            {
                scanf("%d",&h);
                tmp=tmp^sg[h];
            }
            if (tmp)
                putchar('W');
            else
                putchar('L');
        }
        putchar('\n');
        scanf("%d",&k);
    }
    return 0;
}
