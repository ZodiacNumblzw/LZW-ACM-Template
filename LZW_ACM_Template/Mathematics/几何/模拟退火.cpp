//模拟退火算法   POJ2420
int n;
struct Point
{
    double x, y;
    Point(int x=0,int y=0):x(x),y(y) {}
} P[maxn];
double dist(Point A, Point B)
{
    return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y) );
}
int dx[4]= {0,0,1,-1};
int dy[4]= {1,-1,0,0};

struct Simulated_Annealing
{
#define eps 1e-7
#define T 100
#define delta 0.985
//适当调参，delta越小会导致结果越不精准，但是程序，运行速度很快
//delta越大结果越精准，但是运行速度会变慢
#define INF 1e99
    double solve(Point p[],int n)
    {
        //选取初始温度，状态，答案
        Point s = p[0];
        double t = T;
        double ans = INF;
        while(t > eps)
        {
            /******** may need change ************/
            int flag=1;
            while(flag)
            {
                flag=0;
                for(int i=0; i<4; i++)
                {
                    double sum=0;
                    Point pp=Point(s.x+dx[i]*t,s.y+dy[i]*t);
                    for(int j=1; j<=n; j++)
                    {
                        sum+=dist(P[j],pp);
                    }
                    if(sum<ans)
                    {
                        ans=sum;
                        s=pp;
                        flag=1;
                    }
                }
            }

            /**********************************/
            t *= delta;
        }
        return ans;
    }

#undef eps
#undef T
#undef delta
#undef INF
} S;

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    cin>>n;
    for(int i=1; i<=n; i++)
    {
        cin>>P[i].x>>P[i].y;
    }
    cout<<(int)(S.solve(P,n)+0.5)<<endl;
    return Accepted;
}