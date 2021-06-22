//AC自动机板子 by:sjf
struct AC_Automaton
{
    #define MAX 100010
    static const int K=26;//may need change
    int next[MAX][K],fail[MAX],cnt[MAX],last[MAX];
    int root,tot;
    inline int getid(char c)//may need change
    {
        return c-'a';
    }
    int newnode()
    {
        memset(next[tot],0,sizeof(next[tot]));
        fail[tot]=0;
        cnt[tot]=0;
        return tot++;
    }
    void init()
    {
        tot=0;
        root=newnode();
    }
    void insert(char *s)
    {
        int len,now,i;
        len=strlen(s);
        now=root;
        for(i=0;i<len;i++)
        {
            int t=getid(s[i]);
            if(!next[now][t]) next[now][t]=newnode();
            now=next[now][t];
        }
        cnt[now]++;
    }
    void setfail()
    {
        int i,now;
        queue<int>q;
        for(i=0;i<K;i++)
        {
            if(next[root][i]) q.push(next[root][i]);
        }
        while(!q.empty())
        {
            now=q.front();
            q.pop();
            //suffix link
            if(cnt[fail[now]]) last[now]=fail[now];
            else last[now]=last[fail[now]];
            /*
            may need add something here:
            cnt[now]+=cnt[fail[now]];
            */
            for(i=0;i<K;i++)
            {
                if(next[now][i])
                {
                    fail[next[now][i]]=next[fail[now]][i];
                    q.push(next[now][i]);
                }
                else next[now][i]=next[fail[now]][i];
            }
        }
    }
    int query(char *s)
    {
        int len,now,i,res;
        len=strlen(s);
        now=root;
        res=0;
        for(i=0;i<len;i++)
        {
            int t=getid(s[i]);
            now=next[now][t];
            int tmp=now;
            while(tmp&&cnt[tmp]!=-1)
            {
                res+=cnt[tmp];
                cnt[tmp]=-1;
                tmp=last[tmp];
            }
        }
        return res;
    }
    //build fail tree
    vector<int> mp[MAX];
    void build_tree()
    {
        for(int i=0;i<=tot;i++) mp[i].clear();
        for(int i=1;i<tot;i++) mp[fail[i]].pb(i);
    }
    #undef MAX
}ac;







// Created by calabash_boy on 18/6/5.
// HDU 6138
//给定若干字典串。
// query:strx stry 求最长的p,p为strx、stry子串，且p为某字典串的前缀
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+100;
struct Aho_Corasick_Automaton
{
//basic
    int nxt[maxn*10][26],fail[maxn*10];
    int root,tot;
//special
    int flag[maxn*10];
    int len[maxn*10];
    void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        flag[tot] = len[tot]=0;
        return tot;
    }
    void insert(char *s )
    {
        int now = root;
        while (*s)
        {
            int id = *s-'a';
            if(!nxt[now][id])nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now]+1;
            now = nxt[now][id];
        }
    }
    void insert(string str)
    {
        int now = root;
        for (int i=0; i<str.size(); i++)
        {
            int id = str[i]-'a';
            if(!nxt[now][id])nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now]+1;
            now = nxt[now][id];
        }
    }
    void build()
    {
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (int i=0; i<26; i++)
            {
                if(!nxt[head][i])continue;
                int temp = nxt[head][i];
                fail[temp] = fail[head];
                while (fail[temp]&&!nxt[fail[temp]][i])
                {
                    fail[temp] = fail[fail[temp]];
                }
                if(head&&nxt[fail[temp]][i])fail[temp] = nxt[fail[temp]][i];
                Q.push(temp);
            }
        }
    }
    void search(string str,int QID);
    int query(string str,int QID);
} acam;
void Aho_Corasick_Automaton::search(string str,int QID)
{
    int now = root;
    for (int i=0; i<str.size(); i++)
    {
        int id = str[i]-'a';
        now = nxt[now][id];
        int temp = now;
        while (temp!=root&&flag[temp]!=QID)
        {
            flag[temp] = QID;
            temp = fail[temp];
        }
    }
}
int Aho_Corasick_Automaton::query(string str, int QID)
{
    int ans =0;
    int now = root;
    for (int i=0; i<str.size(); i++)
    {
        int id = str[i]-'a';
        now = nxt[now][id];
        int temp = now;
        while (temp!=root)
        {
            if(flag[temp]==QID)
            {
                ans = max(ans,len[temp]);
                break;
            }
            temp = fail[temp];
        }
    }
    return ans;
}
string a[maxn];
int m,n,qid;
int main()
{
    int T;
    cin>>T;
    while (T--)
    {
        acam.clear();
        cin>>n;
        for (int i=1; i<=n; i++)
        {
            cin>>a[i];
            acam.insert(a[i]);
        }
        acam.build();
        cin>>m;
        for (int i=1; i<=m; i++)
        {
            int x,y;
            cin>>x>>y;
            qid++;
            acam.search(a[x],qid);
            int ans = acam.query(a[y],qid);
            cout<<ans<<endl;
        }
    }
    return 0;
}






//AC自动机 by:LZW  HDU2896
#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<vector>
#include<string>
#include<list>
#include<stack>
#include<queue>
#include<deque>
#include<map>
#include<set>
#include<bitset>
#include<utility>
#include<iomanip>
#include<climits>
#include<complex>
#include<cassert>
#include<functional>
#include<numeric>
#define Accepted 0
typedef long long ll;
typedef long double ld;
const int mod=1e9+7;
const int maxn=1e5+10;
const double pi=acos(-1);
const double eps=1e-6;
const int INF=0x3f3f3f3f;
using namespace std;
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}
ll lowbit(ll x)
{
    return x&(-x);
}

struct Aho_Corasick_Automaton
{
#define MAX maxn
#define type int
/***********************/
//basic
    //nxt为构建字典树的数组,fail维护失配时的转移后的结点的下标
    int nxt[MAX][100],fail[MAX];
    int root,tot;        //根节点下标和字典树中结点的数量
//special 结点中维护的信息，和结点同步更新
    type v[MAX];
/***********************/
    int getid(char c)
    {
        return c-32;     //may change
    }
    void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    //从内存池中新建一个结点
    int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        //////
        v[tot]=0;
        return tot;
    }
    //向Trie中插入一个字符串
    void insert(string str,int num)
    {
        int now = root;
        int len=str.size();
        for (int i=0; i<len; i++)
        {
            int id = getid(str[i]);
            if(!nxt[now][id])nxt[now][id] = newnode();
            now = nxt[now][id];
            if(i==len-1)
                v[now]=num;
        }
    }
    //BFS建立fail指针
    void build()
    {
        //root的fail为自己
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (int i=0; i<100; i++)
            {
                if(!nxt[head][i])continue;

                int temp = nxt[head][i];
                fail[temp] = fail[head];
                //若temp的fail没有到达root,且temp当前的fail位置的下一个位置的对应结点为空
                //则temp的fail再向前移动，这里的转移可以结合kmp算法理解
                while (fail[temp]&&!nxt[fail[temp]][i])
                {
                    fail[temp] = fail[fail[temp]];
                }
                //如果temp的fail位置的下一个对应结点存在，则直接赋值
                if(head&&nxt[fail[temp]][i])fail[temp] = nxt[fail[temp]][i];
                Q.push(temp);
            }
        }
    }
    int query(string str,int num);
#undef type
#undef MAX
} ac;

int Aho_Corasick_Automaton::query(string str,int num)
{
    set<int> s;
    int ans = 0;
    int now = root,len=str.length();
    for (int i=0; i<len; i++)
    {
        int id = getid(str[i]);
        while(now&&!nxt[now][id])
        {
            now=fail[now];
        }
        int temp=nxt[now][id];
        while(temp)
        {
            if(v[temp])
                s.insert(v[temp]);
            temp=fail[temp];
        }
        now=nxt[now][id];
    }
    if(s.size()>0)
    {
        cout<<"web "<<num<<':';
        for(int c:s)
        {
            cout<<' '<<c;
        }
        cout<<endl;
        return 1;
    }
    else
        return 0;
}
string str;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int n;
    cin>>n;
    for (int i=1; i<=n; i++)
    {
        cin>>str;
        ac.insert(str,i);
    }
    /*****别忘记build*****/
    ac.build();
    int m;
    cin>>m;
    int ans=0;
    for(int i=1;i<=m;i++)
    {
        cin>>str;
        ans+=ac.query(str,i);
    }
    cout<<"total: "<<ans<<endl;
    return 0;
}

