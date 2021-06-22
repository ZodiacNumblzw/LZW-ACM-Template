//exCRT��չ�й�ʣ�ඨ�� �������inv��Ҫ��exgxd����Ԫ����Ϊ��һ�����ʣ�
ll M[maxn];
ll C[maxn];
void exCRT()
{
    int n;
    scanf("%d",&n);
    for (LL i = 1; i <= n; i++)
        scanf("%lld%lld", &M[i], &C[i]);
    bool flag = 1;
    for (ll i = 2; i <= n; i++)
    {
        ll M1 = M[i - 1], M2 = M[i], C2 = C[i], C1 = C[i - 1], T = gcd(M1, M2);
        if ((C2 - C1) % T != 0)
        {
            flag = 0;
            break;
        }
        M[i] = (M1 * M2) / T;
        C[i] = ( inv( M1 / T, M2 / T ) * (C2 - C1) / T ) % (M2 / T) * M1 + C1;
        C[i] = (C[i] % M[i] + M[i]) % M[i];
    }
    printf("%lld\n", flag ? C[n] : -1);
}