//Trie �ֵ���ģ��
struct Trie
{
    #define type int
    #define MAX maxn*20 //Trie����������� ���ַ�����
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
        sz=1;                     //��¼Trie�н�������
        memset(trie,0,sizeof(trie));
    }
    void insert(const string& s)           //�ڸ��ڵ㴦����һ���ַ���
    {
        int p=1;                         //Ĭ��Ϊ���ڵ�
        for(char c:s)
        {
            int ch=c-'a';
            if(!trie[p].nxt[ch]) trie[p].nxt[ch]=++sz;   //���ڴ����ѡȡһ���ռ������½ڵ�
            p=trie[p].nxt[ch];
            trie[p].pre++;
        }
        trie[p].ed=1;
    }
    bool search(const string& s)        //��ѯTrie���Ƿ����һ���ַ���������1/0
    {
        int p=1;
        for(char c:s)
        {
            p=trie[p].nxt[c-'a'];
            if(!p) return 0;
        }
        return trie[p].ed;
    }
    string prefix(const string& s)     //�����Сuniqueǰ׺
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

//PS:foreachѭ����C++11��ģ�poj������...

//��дһ��
struct Trie
{
    #define type int
    #define MAX maxn*20 //Trie����������� ���ַ�����
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
        sz=1;               //��¼Trie�н�������
        memset(trie,0,sizeof(trie));
    }
    //�ڸ��ڵ㴦����һ���ַ���
    void insert(const string& s)
    {
        int p=1,len=s.length();        //pĬ����ʼΪ���ڵ�
        for(int i=0;i<len;i++)
        {
            int ch=s[i]-'a';
            if(!trie[p].nxt[ch]) trie[p].nxt[ch]=++sz;
            //���ڴ����ѡȡһ���ռ������½ڵ�
            p=trie[p].nxt[ch];
            trie[p].pre++;
        }
        trie[p].ed=1;
    }
    //��ѯTrie���Ƿ����һ���ַ���������1/0
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
    //�����Сuniqueǰ׺
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