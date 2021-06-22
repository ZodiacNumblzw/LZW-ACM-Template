//回文树板子           by:sjf
struct Palindrome_Tree
{
    int len[MAX],next[MAX][26],fail[MAX],last,s[MAX],tot,n;
    int cnt[MAX],deep[MAX];
    int newnode(int l)
    {
        mem(next[tot],0);
        fail[tot]=0; 
        deep[tot]=cnt[tot]=0;
        len[tot]=l;
        return tot++;
    }
    void init()
    {
        tot=n=last=0;
        newnode(0);
        newnode(-1);
        s[0]=-1;
        fail[0]=1;
    }
    int get_fail(int x)
    {
        while(s[n-len[x]-1]!=s[n]) x=fail[x];
        return x;
    }
    void add(int t)//attention the type of t is int
    {
        int id,now;
        s[++n]=t;
        now=get_fail(last);
        if(!next[now][t])
        {
            id=newnode(len[now]+2);
            fail[id]=next[get_fail(fail[now])][t];
            deep[id]=deep[fail[id]]+1;
            next[now][t]=id;
        }
        last=next[now][t];
        cnt[last]++;
    }
    void count()
    {
        for(int i=tot-1;~i;i--) cnt[fail[i]]+=cnt[i];
    }
}pam; //pam.init(); 













//回文树板子              by:calabash_boy
struct Palindromic_AutoMaton{
    //basic
    int s[maxn],now;
    int nxt[maxn][26],fail[maxn],l[maxn],last,tot;
    // extension
    int num[maxn];/*节点代表的所有回文串出现次数*/
    void clear(){
        //1节点：奇数长度root 0节点：偶数长度root
        s[0]=l[1]=-1;
        fail[0] = tot = now =1;
        last = l[0]=0;
        memset(nxt[0],0,sizeof nxt[0]);
        memset(nxt[1],0,sizeof nxt[1]);
    }
    Palindromic_AutoMaton(){clear();}
    int newnode(int ll){
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        fail[tot]=num[tot]=0;
        l[tot]=ll;
        return tot;
    }
    int get_fail(int x){
        while (s[now-l[x]-2]!=s[now-1])x = fail[x];
        return x;
    }
    void add(int ch){
        s[now++] = ch;
        int cur = get_fail(last);
        if(!nxt[cur][ch]){
            int tt = newnode(l[cur]+2);
            fail[tt] = nxt[get_fail(fail[cur])][ch];
            nxt[cur][ch] = tt;
        }
        last = nxt[cur][ch];num[last]++;
    }
    void build(){
        //fail[i]<i，拓扑更新可以单调扫描。
        for (int i=tot;i>=2;i--){
            num[fail[i]]+=num[i];
        }
        num[0]=num[1]=0;
        ans2 -= tot - 1;
    }
    void init(char* ss){
        while (*ss){
            add(*ss-'a');ss++;
        }
    }
    void init(string str){
        for (int i=0;i<str.size();i++){
            add(str[i]-'a');
        }
    }
    long long query();
}pam;
long long Palindromic_AutoMaton::query(){
    long long ret =1;
    for (int i=2;i<=tot;i++){
        ret = max(ret,1LL*l[i]*num[i]);
    }
    return ret;
}
