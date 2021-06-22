//kmp算法    O(m+n)判断字符串匹配/求匹配数
int Next[maxn];
void getNext(const string s)
{
    //k:最大前后缀长度
    int len=s.length();//模版字符串长度
    Next[0] = 0;//模版字符串的第一个字符的最大前后缀长度为0
    for (int i = 1,k = 0; i < len; i++)//for循环，从第二个字符开始，依次计算每一个字符对应的next值
    {
        while(k > 0 && s[i] != s[k])//递归的求出P[0]···P[q]的最大的相同的前后缀长度k
            k = Next[k-1];          //不理解没关系看下面的分析，这个while循环是整段代码的精髓所在，确实不好理解
        if (s[i] == s[k])//如果相等，那么最大相同前后缀长度加1
        {
            k++;
        }
        Next[i] = k;
    }
}

int kmp(const string s,const string p)
{
    int cnt=0;
    int len1=s.length(),len2=p.length();
    getNext(p);
    for (int i = 0,j = 0; i < len1; i++)
    {
        while(j > 0 && s[i] != p[j])   //失配，j下标向前挪找到可以匹配的位置
            j = Next[j-1];
        if (s[i] == p[j])//该位匹配成功，j下标挪到模式串的下一个位置
        {
            j++;
        }
        if (j == len2)            //匹配完毕
        {
            cnt++;
            j=Next[j-1];
            //break;
        }
    }
    return cnt;
}
