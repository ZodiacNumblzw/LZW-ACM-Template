//指针版Treap 省内存
// It is made by XZZ
#include<cstdio>
#include<algorithm>
#define pr pair<point,point>
#define mp make_pair
using namespace std;
#define rep(a,b,c) for(rg int a=b;a<=c;a++)
#define drep(a,b,c) for(rg int a=b;a>=c;a--)
#define erep(a,b) for(rg int a=fir[b];a;a=nxt[a])
#define il inline
#define rg register
#define vd void
typedef long long ll;
il int gi(){
    rg int x=0,f=1;rg char ch=getchar();
    while(ch<'0'||ch>'9')f=ch=='-'?-1:f,ch=getchar();
    while(ch>='0'&&ch<='9')x=x*10+ch-'0',ch=getchar();
    return x*f;
}
int seed=19260817;
il int Rand(){return seed=seed*48271ll%2147483647;}
typedef struct node* point;
point null;
struct node{
    char data;
    int size,rand;
    point ls,rs;
    bool rev;
    node(char ch){data=ch,size=1,rand=Rand(),rev=0,ls=rs=null;}
    il vd down(){if(rev)rev=0,ls->rev^=1,rs->rev^=1,swap(ls,rs);}
    il vd reset(){if(ls!=null)ls->down();if(rs!=null)rs->down();size=ls->size+rs->size+1;}
};
point root=null;
il point build(int n){
    point stack[n+1];
    int top=0;
    char ch;
    rep(i,1,n){
    ch=getchar();while(ch=='\n')ch=getchar();
    point now=new node(ch),lst=null;
    while(top&&stack[top]->rand>now->rand)lst=stack[top],stack[top--]->reset();
    now->ls=lst;if(top)stack[top]->rs=now;stack[++top]=now;
    }
    while(top)stack[top--]->reset();
    return stack[1];
}
il point merge(point a,point b){
    if(a==null)return b;
    if(b==null)return a;
    if(a->rand<b->rand){a->down(),a->rs=merge(a->rs,b),a->reset();return a;}
    else {b->down(),b->ls=merge(a,b->ls),b->reset();return b;}
}
il pr split(point now,int num){
    if(now==null)return mp(null,null);
    now->down();
    point ls=now->ls,rs=now->rs;
    if(num==ls->size){now->ls=null,now->reset();return mp(ls,now);}
    if(num==ls->size+1){now->rs=null,now->reset();return mp(now,rs);}
    if(num<ls->size){pr T=split(ls,num);now->ls=T.second,now->reset();return mp(T.first,now);}
    pr T=split(rs,num-ls->size-1);now->rs=T.first,now->reset();return mp(now,T.second);
}
il vd del(point now){if(now!=null)del(now->ls),del(now->rs),delete now;}
int main(){
    int m=gi()-1,pos=0;
    char opt[10];
    null=new node('%');
    null->size=0;
    {
    scanf("%*s");
    root=build(gi());
    }
    while(m--){
    scanf("%s",opt);
    if(opt[0]=='M')pos=max(0,min(gi(),root->size));
    else if(opt[0]=='P'){if(pos)--pos;}
    else if(opt[0]=='N'){if(pos!=root->size)++pos;}
    else if(opt[0]=='G'){
        pr T=split(root,pos);
        char lst;point now=T.second;
        while(now!=null)lst=now->data,now->down(),now=now->ls;
        printf("%c\n",lst);
        root=merge(T.first,T.second);
    }
    else if(opt[0]=='I'){
        pr T=split(root,pos);
        root=merge(T.first,merge(build(gi()),T.second));
    }
    else{
        pr T=split(root,pos),TT=split(T.second,gi());
        if(opt[0]=='D')root=merge(T.first,TT.second),del(TT.first);
        else TT.first->rev^=1,root=merge(T.first,merge(TT.first,TT.second));
    }
    }
    del(root);
    delete null;
    return 0;
}







//Treap
struct node{ //节点数据的结构
    int key,prio,size; //size是指以这个节点为根的子树中节点的数量
    node* ch[2]; //ch[0]指左儿子，ch[1]指右儿子
};

typedef node* tree;

node base[MAXN],nil;
tree top,null,root;

void init(){                //初始化top和null
    top=base;
    root=null=&nil;
    null->ch[0]=null->ch[1]=null;
    null->key=null->prio=2147483647;
    null->size=0;
}

inline tree newnode(int k){ //注意这种分配内存的方法也就比赛的时候用用，仅仅是为了提高效率
    top->key=k;				//看为val,是BST的键值
    top->size=1;
    top->prio=random();
    top->ch[0]=top->ch[1]=null;
    return top++;
}


//Treap结点的旋转
void rotate(tree &x,bool d){ //d指旋转的方向，0为左旋，1为右旋
    tree y=x->ch[!d];        //x为要旋的子树的根节点
    x->ch[!d]=y->ch[d];
    y->ch[d]=x;
    x->size=x->ch[0]->size+1+x->ch[1]->size;
    y->size=y->ch[0]->size+1+y->ch[1]->size;
    x=y;
}

void insert(tree &t,int key){ //插入一个节点
    if (t==null) t=newnode(key);
    else{
        bool d=key>t->key;
        insert(t->ch[d],key);
        t->size++;
        if (t->prio<t->ch[d]->prio) rotate(t,!d);
    }
}

void erase(tree &t,int key){ //删除一个节点
    if (t->key!=key){
        erase(t->ch[key>t->key],key);
        t->size--;
	}
    else if (t->ch[0]==null) t=t->ch[1];
    else if (t->ch[1]==null) t=t->ch[0];
    else{
        bool d=t->ch[0]->prio<t->ch[1]->prio;
        rotate(t,d);
        erase(t->ch[d],key);
    }
}

tree select(int k){ //选择第k小节点
    tree t=root;
    for (int tmp;;){
        tmp=t->ch[0]->size+1;
        if (k==tmp) return t;
        if (k>tmp){
            k-=tmp;
            t=t->ch[1];
        }
        else t=t->ch[0];
    }
}
