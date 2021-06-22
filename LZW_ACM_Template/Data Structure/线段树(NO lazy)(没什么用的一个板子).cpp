//�߶��� (NO lazy)
//[l,r]����Ϊһ�����������Ϊ���rt��ָ��������
int a[maxn],n;
int Sum[maxn<<2];
void update(int index)
{
    Sum[index]=Sum[index<<1]+Sum[index<<1|1];               //��Ϊ����͵�ģ�壬������������ϲ������޸����ڴ��޸�
}
void build(int l,int r,int rt)
{
    if(l==r)
    {
        Sum[rt]=a[l];
        return;
    }
    int m=(l+r)>>1;
    build(l,m,rt<<1);
    build(m+1,r,rt<<1|1);
    update(rt);
}
int ans_ans;
void ask(int A,int l,int r,int rt)     //�����ѯ
{
    if(l==r)
    {
        ans_ans=Sum[rt];
        return;
    }
    int m=(l+r)>>1;
    if(A<=m)
        ask(A,l,m,rt<<1);
    else
        ask(A,m+1,r,rt<<1|1);
}

void add(int L,int C,int l,int r,int rt)          //�����޸�
{
    if(l==r)
    {
        Sum[rt]+=C;
        return ;
    }
    int m=(l+r)>>1;
    if(L<=m)
        add(L,C,l,m,rt<<1);
    else
        add(L,C,m+1,r,rt<<1|1);
    update(rt);
}
//�������[L,R]�����ı�
int Query(int L,int R,int l,int r,int rt)          //�������
{
    if(l>=L&&r<=R)
    {
        return Sum[rt];
    }
    int m=(l+r)>>1;
    int ans=0;
    if(L<=m)
        ans+=Query(L,R,l,m,rt<<1);
    if(R>m)
        ans+=Query(L,R,m+1,r,rt<<1|1);
    return ans;
} 