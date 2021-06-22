//kmp�㷨    O(m+n)�ж��ַ���ƥ��/��ƥ����
int Next[maxn];
void getNext(const string s)
{
    //k:���ǰ��׺����
    int len=s.length();//ģ���ַ�������
    Next[0] = 0;//ģ���ַ����ĵ�һ���ַ������ǰ��׺����Ϊ0
    for (int i = 1,k = 0; i < len; i++)//forѭ�����ӵڶ����ַ���ʼ�����μ���ÿһ���ַ���Ӧ��nextֵ
    {
        while(k > 0 && s[i] != s[k])//�ݹ�����P[0]������P[q]��������ͬ��ǰ��׺����k
            k = Next[k-1];          //�����û��ϵ������ķ��������whileѭ�������δ���ľ������ڣ�ȷʵ�������
        if (s[i] == s[k])//�����ȣ���ô�����ͬǰ��׺���ȼ�1
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
        while(j > 0 && s[i] != p[j])   //ʧ�䣬j�±���ǰŲ�ҵ�����ƥ���λ��
            j = Next[j-1];
        if (s[i] == p[j])//��λƥ��ɹ���j�±�Ų��ģʽ������һ��λ��
        {
            j++;
        }
        if (j == len2)            //ƥ�����
        {
            cnt++;
            j=Next[j-1];
            //break;
        }
    }
    return cnt;
}
