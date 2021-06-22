
//���������Ż�dp   �����P3957�����ӣ�
int a[maxn];
int s[maxn];
ll dp[maxn];
ll k;
int n,d,ans=-1;
int q[maxn];        //q������¼�±꣬��q[l]~q[r]�ڵ��±궼�ǺϷ���
bool check(int g)
{
    //cout<<"g="<<g<<endl;
    for(int i=1; i<=n; i++)
    {
        dp[i]=-1e18;
    }
    int x1=g+d,x2=max(d-g,1);
    int now=0;
    int l=0,r=-1;     //[l,r]��¼���е����� ��ʼ״̬l>r����Ϊ��
    int i;
    for(i=1; i<=n; i++)
    {
        int L=a[i]-x1,R=a[i]-x2; //L,R������¼�Ϸ���Χ
        //cout<<L<<' '<<R<<endl;
        while(a[now]<=R&&now<i)    //Insert(now)�����
        {
            while(r>=l&&dp[q[r]]<=dp[now])r--;   //ά�����ֵ���ͱ�֤��һ�������ݼ��ĵ�������
            q[++r]=now;
            now++;
        }
        while(a[q[l]]<L&&l<=r)l++;            //pop_front �Ѷ���ǰ�˲��Ϸ�����ȥ��
        if(l>r||dp[q[l]]==-1e18)
            continue;
        dp[i]=dp[q[l]]+s[i];
        //cout<<a[i]<<' '<<dp[i]<<endl;
        if(dp[i]>=k)
            return true;
    }
    return false;
}




//���������Ż�dp (HDU����)
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
    //top�Ƕ��׽�㣬tailΪ��β���
    int top,tail;
    Clique(){top=1,tail=0;}
    void Init() {top=1,tail=0;}
    //����һ�����ʱ,ɾ�����ֲ��ŵĽ��
    void push(long long f,long long s)
    {
        while (tail>=top && s>q[tail].second)
            tail--;
        ++tail;
        q[tail].first=f;
        q[tail].second=s;
    }
    //�ڲ�ѯ֮ǰ�������Ѿ�Խ��Ĳ�����Ҫ��Ľ��
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