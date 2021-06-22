//Trie 字典树模板
struct Trie
{
    #define type int
    #define MAX maxn*20 //Trie结点的最大数量 总字符数量
    int sz;
    struct TrieNode
    {
        int pre;
        bool ed;
        int nxt[26];
        type v;                //may change
    }trie[MAX];
    Trie()
    {
        sz=1;                     //记录Trie中结点的数量
        memset(trie,0,sizeof(trie));
    }
    void insert(const string& s)           //在根节点处插入一个字符串
    {
        int p=1;                         //默认为根节点
        for(char c:s)
        {
            int ch=c-'a';
            if(!trie[p].nxt[ch]) trie[p].nxt[ch]=++sz;   //从内存池中选取一个空间分配给新节点
            p=trie[p].nxt[ch];
            trie[p].pre++;
        }
        trie[p].ed=1;
    }
    bool search(const string& s)        //查询Trie中是否存在一个字符串，返回1/0
    {
        int p=1;
        for(char c:s)
        {
            p=trie[p].nxt[c-'a'];
            if(!p) return 0;
        }
        return trie[p].ed;
    }
    string prefix(const string& s)     //求出最小unique前缀
    {
        string res;
        int p=1;
        for(char c:s)
        {
            p=trie[p].nxt[c-'a'];
            res+=c;
            if(trie[p].pre<=1) break;
        }
        return res;
    }
    #undef type
}tr;

//PS:foreach循环是C++11里的，poj不能用...

//再写一遍
struct Trie
{
    #define type int
    #define MAX maxn*20 //Trie结点的最大数量 总字符数量
    int sz;
    struct TrieNode
    {
        int pre;
        bool ed;
        int nxt[26];
        type v;            //may change
    }trie[MAX];
    Trie()
    {
        sz=1;               //记录Trie中结点的数量
        memset(trie,0,sizeof(trie));
    }
    //在根节点处插入一个字符串
    void insert(const string& s)
    {
        int p=1,len=s.length();        //p默认起始为根节点
        for(int i=0;i<len;i++)
        {
            int ch=s[i]-'a';
            if(!trie[p].nxt[ch]) trie[p].nxt[ch]=++sz;
            //从内存池中选取一个空间分配给新节点
            p=trie[p].nxt[ch];
            trie[p].pre++;
        }
        trie[p].ed=1;
    }
    //查询Trie中是否存在一个字符串，返回1/0
    bool search(const string& s)
    {
        int p=1,len=s.length();
        for(int i=0;i<len;i++)
        {
            p=trie[p].nxt[s[i]-'a'];
            if(!p) return 0;
        }
        return trie[p].ed;
    }
    //求出最小unique前缀
    string prefix(const string& s)
    {
        string res;
        int p=1,len=s.length();
        for(int i=0;i<len;i++)
        {
            p=trie[p].nxt[s[i]-'a'];
            res+=s[i];
            if(trie[p].pre<=1) break;
        }
        return res;
    }
    #undef type
}tr;