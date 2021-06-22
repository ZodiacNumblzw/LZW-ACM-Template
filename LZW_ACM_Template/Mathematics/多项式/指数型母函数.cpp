//指数型母函数
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>
 
using namespace std;
 
typedef long long ll;
 
const int maxn = 1e5 + 10;
#define mem(a) memset(a, 0, sizeof a)
 
double a[maxn],b[maxn]; // 注意为浮点型
 
int s1[maxn];
 
double f[11];
void init() {
    mem(a);
    mem(b);
    mem(s1);
    f[0] = 1;
    for (int i = 1; i <= 10; i++) {
        f[i] = f[i - 1] * i;
    }
}
 
int main() {
    int n,m;
    while (~scanf("%d%d", &n, &m)) {
       init();
       for (int i = 0; i < n; i++) {
            scanf("%d", &s1[i]);
       }
        for (int i = 0; i <= s1[0]; i++) a[i] = 1.0 / f[i];
        for (int i = 1; i < n; i++) {
            mem(b);
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= s1[i] && k + j <= m; k++) {
                    b[j + k] += a[j] * 1.0 / f[k]; //注意这里
                }
            }
            memcpy(a, b, sizeof b);
        }
       printf("%.0f\n", a[m] * f[m]);
    }
    return 0;
}







#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
typedef long long ll;
using namespace std;
const int N = 11;
double c1[N],c2[N];     //注意类型
ll fac[N];
//预处理阶乘
void cal()
{
    fac[0]=1;           //0!会被用到
    for(int i=1; i<N; i++)
        fac[i]=i*fac[i-1];
}

int main()
{
    cal();
    int n,r;
    while(~scanf("%d%d",&n,&r))
    {
        memset(c1,0,sizeof(c1));
        memset(c2,0,sizeof(c2));

        c1[0]=1;
        int num;
        //计算多项式的乘积
        for(int i=1; i<=n; i++)
        {
            scanf("%d",&num);
            if(num==0) continue;
            for(int j=0; j<=r; j++)
            {
                for(int k=0; k<=num&&k+j<=r; k++)
                {
                    c2[k+j]+=c1[j]/fac[k];
                }
            }
            for(int j=0; j<=r; j++)
            {
                c1[j]=c2[j];
                c2[j]=0;
            }
        }

        printf("%lld\n",(ll)(c1[r]*fac[r]+0.5));
    }

    return 0;
}
