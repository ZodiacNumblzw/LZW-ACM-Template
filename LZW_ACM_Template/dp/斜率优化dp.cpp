//斜率优化dp板子
struct Line{
    ll k,b;
    ll f(ll x){
        return k*x+b;
    }
};
struct Hull{
    vector<Line>ve;
    int cnt,idx;
    bool empty(){return cnt==0;}
    void init(){ve.clear();cnt=idx=0;}
    void add(const Line& p){ve.push_back(p);cnt++;}
    void pop(){ve.pop_back();cnt--;}
    bool checkld(const Line& a,const Line& b,const Line& c){
        return (long double)(a.b-b.b)/(long double)(b.k-a.k)>(long double)(a.b-c.b)/(long double)(c.k-a.k);
    }
    bool checkll(const Line& a,const Line& b,const Line& c){
        return (a.b-b.b)*(c.k-a.k)>(a.b-c.b)*(b.k-a.k);
    }
    void insert(const Line& p){
        if(cnt&&ve.back().k==p.k){
            if(p.b<=ve.back().b)return;
            else pop();
        }
        while(cnt>=2&&checkld(ve[cnt-2],ve[cnt-1],p))pop();
        add(p);
    }
    ll query(ll x){
        while(idx+1<cnt&&ve[idx+1].f(x)>ve[idx].f(x))idx++;
        return ve[idx].f(x);
    }
}hull[2005];

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41129697





//斜率优化dp  quailty
struct Line {
    mutable ll k,m,p;
    bool operator <(const Line& o)const { return k<o.k; }
    bool operator <(ll x)const { return p<x; }
};
struct LineContainer : multiset<Line,less<> > {
    const ll inf = LLONG_MAX;
    ll div(ll a,ll b){
        return a/b-((a^b)<0&&a%b);
    }
    bool isect(iterator x,iterator y){
        if (y==end()){x->p=inf; return false; }
        if (x->k==y->k)x->p=x->m>y->m?inf:-inf;
        else x->p=div(y->m-x->m,x->k-y->k);
        return x->p>=y->p;
    }
    void add(ll k,ll m) {
        auto z=insert({k,m,0}),y=z++,x=y;
        while (isect(y,z))z=erase(z);
        if (x!=begin() && isect(--x,y))isect(x,y=erase(y));
        while ((y=x)!=begin() && (--x)->p>=y->p)
            isect(x,erase(y));
    }
    ll query(ll x){
        assert(!empty());
        auto l=*lower_bound(x);
        return l.k*x+l.m;
    }
}h;

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41130719
