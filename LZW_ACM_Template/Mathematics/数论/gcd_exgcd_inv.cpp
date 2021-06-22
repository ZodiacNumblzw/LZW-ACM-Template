//gcd
inline ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
//exgcd
ll exgcd(ll a, ll b, ll &x, ll &y) {     //ax+by=gcd(a,b)
    if (b == 0) {x = 1, y = 0; return a;}
    ll r = exgcd(b, a % b, x, y), tmp;
    tmp = x; x = y; y = tmp - (a / b) * y;
    return r;
}

int exgcd(int a,int b) //ax+by=gcd(a,b) return x;  非递归写法
{
    int x0=1,y0=0,x1=0,y1=1,x=a,y=b,r=a%b,q=a/b;
    while(r)
    {
        x=x0-q*x1;
        y=y0-q*y1;
        x0=x1,y0=y1;
        x1=x,y1=y;
        a=b;
        b=r;
        r=a%b;
        q=a/b;
    }
    return x;
}



//求逆元
ll inv(ll a, ll b) {
    ll r = exgcd(a, b, x, y);
    while (x < 0) x += b;
    return x;
}


//Python板子


/*def gcd(a,b):
    if b==0:
        return a
    else:
        return gcd(b,a%b)
 
 
def exgcd(r0, r1): # calc ax+by = gcd(a, b) return x
    x0, y0 = 1, 0
    x1, y1 = 0, 1
    x, y = r0, r1
    r = r0 % r1
    q = r0 // r1
    while r:
        x, y = x0 - q * x1, y0 - q * y1
        x0, y0 = x1, y1
        x1, y1 = x, y
        r0 = r1
        r1 = r
        r = r0 % r1
        q = r0 // r1
    return x
 
def inv(a,b):
    x=exgcd(a,b)
    while x<0:
        x+=b
    return x
*/