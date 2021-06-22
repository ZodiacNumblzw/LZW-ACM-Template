//后缀自动机板子                     //by Calabash_boy
/*注意需要按l将节点基数排序来拓扑更新parent树*/
struct Suffix_Automaton{
    //basic
    int nxt[maxn*2][26],fa[maxn*2],l[maxn*2];
    int last,cnt;
    int flag[maxn*2];
    Suffix_Automaton(){ clear(); }
    void clear(){
        last =cnt=1;
        fa[1]=l[1]=0;
        memset(nxt[1],0,sizeof nxt[1]);
    }
    void init(char *s){
        while (*s){
            add(*s-'a');s++;
        }
    }
    void build(){
        int temp = 1;
        for (int i=0;i<n;i++){
            temp = nxt[temp][s[i] - 'a'];
            int now = temp;
            while (now && (flag[now] & 2) == 0){
                flag[now] |= 2;
                now = fa[now];
            }
        }
        temp = 1;
        for (int i=0;i<n;i++){
            temp = nxt[temp][t[i] - 'a'];
            int now = temp;
            while (now && (flag[now] & 1) == 0){
                flag[now] |= 1;
                now = fa[now];
            }
        }
        for (int i=1;i<=cnt;i++){
            if (flag[i] == 3){
                ans2 += l[i] - l[fa[i]];
            }
            if (flag[i] & 1){
                ans += l[i] - l[fa[i]];
            }
        }
    }
    void add(int c){
        int p = last;
        int np = ++cnt;
        memset(nxt[cnt],0,sizeof nxt[cnt]);
        l[np] = l[p]+1;last = np;
        while (p&&!nxt[p][c])nxt[p][c] = np,p = fa[p];
        if (!p)fa[np]=1;
        else{
            int q = nxt[p][c];
            if (l[q]==l[p]+1)fa[np] =q;
            else{
                int nq = ++ cnt;
                l[nq] = l[p]+1;
                memcpy(nxt[nq],nxt[q],sizeof (nxt[q]));
                fa[nq] =fa[q];fa[np] = fa[q] =nq;
                while (nxt[p][c]==q)nxt[p][c] =nq,p = fa[p];
            }
        }
    }
 
}sam;