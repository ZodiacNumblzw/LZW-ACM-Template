//Treap部分
//PS:按size来split可以达到区间操作，按value可达到集合操作

//不重复随机数生成函数               (Treap)
inline int Rand(){
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}

//还是随机数生成
int Seed = 19260817 ;
inline int Rand() {
    Seed ^= Seed << 7 ;
    Seed ^= Seed >> 5 ;
    Seed ^= Seed << 13 ;
    return Seed ;
}


//无旋Treap              （普通平衡树）
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-9;
const int maxn=1e5+10;
#define Accepted 0
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
#define ls(x) arr[x].child[0]
#define rs(x) arr[x].child[1]
inline int Rand(){                                 //随机数
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node                                     //Treap结点
{
    int child[2],value,key,size;
}arr[maxn];
int tot;                                       //结点数量
inline void Push_Up(int x)                       //更新size
{
    arr[x].size=arr[ls(x)].size+arr[rs(x)].size+1;
}
void Split(int root,int& x,int& y,int value)         //按值切分无旋Treap树
{
    if(!root)
    {
        x=y=0;
        return ;
    }
    if(arr[root].value<=value) x=root,Split(rs(root),rs(x),y,value);
    else y=root,Split(ls(root),x,ls(y),value);
    Push_Up(root);
}
void Merge(int& root,int x,int y)                   //合并两个无旋Treap子树
{
    if(!x||!y)
    {
        root=x+y;
        return ;
    }
    if(arr[x].key<arr[y].key)root=x,Merge(rs(root),rs(x),y);
    else root=y,Merge(ls(root),x,ls(y));
    Push_Up(root);
}
void Insert(int& root,int value)                  //插入结点
{
    int x=0,y=0,z=++tot;
    arr[z].key=Rand(),arr[z].size=1,arr[z].value=value;
    Split(root,x,y,value);
    Merge(x,x,z);
    Merge(root,x,y);
}
void Erase(int& root,int value)                //删除结点
{
    int x=0,y=0,z=0;
    Split(root,x,y,value);
    Split(x,x,z,value-1);
    Merge(z,ls(z),rs(z));
    Merge(x,x,z);
    Merge(root,x,y);
}
int Kth_number(int root,int k)               //树上第k小
{
    while(arr[ls(root)].size+1!=k)
    {
        if(arr[ls(root)].size>=k)root=ls(root);
        else k-=arr[ls(root)].size+1,root=rs(root);
    }
    return arr[root].value;
}
int Get_rank(int& root,int value)             //树上名次
{
    int x=0,y=0;
    Split(root,x,y,value-1);
    int res=arr[x].size+1;
    Merge(root,x,y);
    return res;
}
int Pre(int& root,int value)                //value的前驱
{
    int x=0,y=0;
    Split(root,x,y,value-1);
    int res=Kth_number(x,arr[x].size);
    Merge(root,x,y);
    return res;
}
int Suf(int& root,int value)               //value的后继
{
    int x=0,y=0;
    Split(root,x,y,value);
    int res=Kth_number(y,1);
    Merge(root,x,y);
    return res;
}
int root;                                //root=0,初始化空树
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int n;
    cin>>n;
    while(n--)
    {
        int opt,x;
        cin>>opt>>x;
        if(opt==1)
            Insert(root,x);
        else if(opt==2)
            Erase(root,x);
        else if(opt==3)
            cout<<Get_rank(root,x)<<endl;
        else if(opt==4)
            cout<<Kth_number(root,x)<<endl;
        else if(opt==5)
            cout<<Pre(root,x)<<endl;
        else if(opt==6)
            cout<<Suf(root,x)<<endl;
    }
    return Accepted;
}






//文艺平衡树 无旋Treap实现
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-9;
const int maxn=1e5+10;
#define Accepted 0
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}
#define ls(x) arr[x].child[0]
#define rs(x) arr[x].child[1]
inline int Rand()                                  //随机数
{
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node                                     //Treap结点
{
    int child[2],value,key,size,tag;
} arr[maxn];
int tot;                                       //结点数量
inline void Push_Up(int x)
{
    arr[x].size=arr[ls(x)].size+arr[rs(x)].size+1;
}

inline void Push_Down(int x)                  //下推翻转标记
{
    if(arr[x].tag)
    {
        arr[ls(x)].tag^=1;
        arr[rs(x)].tag^=1;
        swap(ls(x),rs(x));
        arr[x].tag^=1;
    }
}

void Split(int root,int &x,int &y,int Sz)           //按Size切分
{
    if(!root)
    {
        x=y=0;
        return ;
    }
    Push_Down(root);
    if(Sz<=arr[ls(root)].size)
        y=root,Split(ls(root),x,ls(y),Sz);
    else x=root,Split(rs(root),rs(root),y,Sz-arr[ls(root)].size-1);
    Push_Up(root);
}

void Merge(int&root,int x,int y)             //合并子树
{
    if(!x||!y)
    {
        root=x+y;
        return;
    }
    Push_Down(x),Push_Down(y);
    if(arr[x].key<arr[y].key)root=x,Merge(rs(root),rs(x),y);
    else root=y,Merge(ls(root),x,ls(y));
    Push_Up(root);
}

inline void Insert(int&root,int value)          //插入一个结点
{
    int x=0,y=0,z=++tot;
    arr[z].size=1,arr[z].value=value,arr[z].key=Rand(),arr[z].tag=0;
    Split(root,x,y,value);
    Merge(x,x,z);
    Merge(root,x,y);
}

void Rever(int&root,int L,int R)                 //翻转一个区间
{
    int x=0,y=0,z=0,t=0;
    Split(root,x,y,R);
    Split(x,z,t,L-1);
    arr[t].tag^=1;
    Merge(x,z,t);
    Merge(root,x,y);
}
void Print(int x)                             //中序遍历输出
{
    if(!x)return ;
    Push_Down(x);
    Print(ls(x));
    cout<<arr[x].value<<' ';
    Print(rs(x));
}
int root;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int n,m;
    cin>>n>>m;
    for(int i=1; i<=n; i++)
        Insert(root,i);
    while(m--)
    {
        int a,b;
        cin>>a>>b;
        Rever(root,a,b);
    }
    Print(root);
    return Accepted;
}