//求(a+b*sqrt(x))^n的展开式
//将a+b*sqrt(x)看做是复数 重载乘法，再用快速幂
struct node
{
    int a,b,x;
    node(int a=0,int b=0,int x=0):a(a),b(b),x(x){}
    //(a+b*sqrt(x))*(c+d*sqrt(x))=(ac+bdx)+(ad+bc)*sqrt(x);
    node operator*(const node n)
    {
        return node(a*n.a+x*b*n.b,a*n.b+b*n.a,x);
    }
    void print()
    {
        cout<<a<<'+'<<b<<"sqrt("<<x<<')'<<endl;
    }
};
node quick(node a,int b)
{
    node sum=node(1,0,a.x);
    while(b)
    {
        if(b&1)
        {
            sum=sum*a;
        }
        b>>=1;
        a=(a*a);
    }
    return sum;
}
signed main()
{
    int a,b,x,n;
    while(cin>>a>>b>>x>>n)
    {
        quick(node(a,b,x),n).print();
    }
    return Accepted;
}