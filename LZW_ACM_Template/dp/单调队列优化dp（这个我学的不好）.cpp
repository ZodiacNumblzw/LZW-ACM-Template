
//单调队列优化dp   （洛谷P3957跳房子）
int a[maxn];
int s[maxn];
ll dp[maxn];
ll k;
int n,d,ans=-1;
int q[maxn];        //q用来记录下标，在q[l]~q[r]内的下标都是合法的
bool check(int g)
{
    //cout<<"g="<<g<<endl;
    for(int i=1; i<=n; i++)
    {
        dp[i]=-1e18;
    }
    int x1=g+d,x2=max(d-g,1);
    int now=0;
    int l=0,r=-1;     //[l,r]记录队列的区间 初始状态l>r队列为空
    int i;
    for(i=1; i<=n; i++)
    {
        int L=a[i]-x1,R=a[i]-x2; //L,R用来记录合法范围
        //cout<<L<<' '<<R<<endl;
        while(a[now]<=R&&now<i)    //Insert(now)这个点
        {
            while(r>=l&&dp[q[r]]<=dp[now])r--;   //维护最大值，就保证是一个单调递减的单调队列
            q[++r]=now;
            now++;
        }
        while(a[q[l]]<L&&l<=r)l++;            //pop_front 把队列前端不合法的数去掉
        if(l>r||dp[q[l]]==-1e18)
            continue;
        dp[i]=dp[q[l]]+s[i];
        //cout<<a[i]<<' '<<dp[i]<<endl;
        if(dp[i]>=k)
            return true;
    }
    return false;
}




//单调队列优化dp (HDU板子)
typedef pair<long long,long long> P;
const int maxn=5010,maxk=2010;
 
struct Clique
{
    P q[maxn];
    int top,tail;
 
    void Init() {top=1,tail=0;}
 
    void push(long long k,long long b)
    {
        while (tail>top && (__int128)(b-q[tail-1].second)*(q[tail].first-q[tail-1].first)>=(__int128)(q[tail].second-q[tail-1].second)*(k-q[tail-1].first)) tail--;
        ++tail;
        q[tail].first=k;
        q[tail].second=b;
    }
 
    void autopop(long long x)
    {
        while (top<tail && q[top].first*x+q[top].second<=q[top+1].first*x+q[top+1].second) top++;
    }
 
    P front() {return q[top];}
 
}Q[maxk];

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41136357








struct Clique
{
    #define P pair<long long,long long>
    P q[maxn];
    //top是队首结点，tail为队尾结点
    int top,tail;
    Clique(){top=1,tail=0;}
    void Init() {top=1,tail=0;}
    //加入一个结点时,删除部分不优的结点
    void push(long long f,long long s)
    {
        while (tail>=top && s>q[tail].second)
            tail--;
        ++tail;
        q[tail].first=f;
        q[tail].second=s;
    }
    //在查询之前，弹出已经越界的不符合要求的结点
    void autopop(long long x)
    {
        while (top<=tail && q[top].first<=x) top++;
    }
    ll query(ll x)
    {
        int l=top,r=tail,ans=-1,mid;
        while(l<=r)
        {
            mid=(l+r)>>1;
            if(q[mid].second>=x)
            {
                ans=q[mid].first;
                l=mid+1;
            }
            else
                r=mid-1;
        }
        return ans;
    }
    P front() {return q[top];}
    #undef P
}Q;