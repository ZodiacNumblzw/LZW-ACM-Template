//SG����ģ��        Hdu1536 ���ǿ��Ի���ΪImpartial Combinatorial Games�ĳ���ģ�ͣ�����ͼ�ƶ����ӣ�
//�����Գ�������SG���� ���Ӷ�Ŀǰ����̫����������ǳ���
//���������ÿ�ѵ�������̫�󣬿��Կ��Ǵ��¹�����ѭ����   ��Codeforces1194D
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

//ע�� S����Ҫ����С�������� SG����Ҫ��ʼ��Ϊ-1 ����ÿ������ֻ���ʼ��1��
//����Ҫÿ��һ������SG�ͳ�ʼ��һ��
int SG[10100],n,m,s[102],k;//k�Ǽ���s�Ĵ�С S[i]�Ƕ��������ȡ�����������
int dfs(int x)//��SG[x]ģ��
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

//����SG����
//�������ѵİ���Ҳͦ����ģ���������    
//https://www.cnblogs.com/dyllove98/p/3194312.html

#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;
//ע�� S����Ҫ����С�������� SG����Ҫ��ʼ��Ϊ-1 ����ÿ������ֻ���ʼ��1��
//����Ҫÿ��һ������SG�ͳ�ʼ��һ��
int SG[10100],n,m,s[102],k;//k�Ǽ���s�Ĵ�С S[i]�Ƕ��������ȡ�����������
int dfs(int x)//��SG[x]ģ��
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


//�����Ƕ�SG��������
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
const int K=101;
const int H=10001;//H������Ҫ���򵽵����ֵ
int k,m,l,h,s[K],sg[H],mex[K];///k�Ǽ���Ԫ�صĸ��� s[]�Ǽ���  mex��С��Լ�ͼ��ϴ�С���
///ע��s������
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
        sort(s+1,s+k+1);            //���������
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
