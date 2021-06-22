//��ά����
const double eps = 1e-8;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0,double _y = 0,double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
	void output(){
		printf("%.2lf %.2lf %.2lf\n",x,y,z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	bool operator <(const Point3 &b)const{
		return sgn(x-b.x)==0?(sgn(y-b.y)==0?sgn(z-b.z)<0:y<b.y):x<b.x;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//���
	double operator *(const Point3 &b)const{
		return x*b.x+y*b.y+z*b.z;
	}
	//���
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
	double rad(Point3 a,Point3 b){
		Point3 p = (*this);
		return acos( ( (a-p)*(b-p) )/ (a.distance(p)*b.distance(p)) );
	}
	//�任����
	Point3 trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point3(x*r,y*r,z*r);
	}
};
struct Line3
{
	Point3 s,e;
	Line3(){}
	Line3(Point3 _s,Point3 _e)
	{
		s = _s;
		e = _e;
	}
	bool operator ==(const Line3 v)
	{
		return (s==v.s)&&(e==v.e);
	}
	void input()
	{
		s.input();
		e.input();
	}
	double length()
	{
		return s.distance(e);
	}
	//�㵽ֱ�߾���
	double dispointtoline(Point3 p)
	{
		return ((e-s)^(p-s)).len()/s.distance(e);
	}
	//�㵽�߶ξ���
	double dispointtoseg(Point3 p)
	{
		if(sgn((p-s)*(e-s)) < 0 || sgn((p-e)*(s-e)) < 0)
			return min(p.distance(s),e.distance(p));
		return dispointtoline(p);
	}
	//`���ص�p��ֱ���ϵ�ͶӰ`
	Point3 lineprog(Point3 p)
	{
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`p�ƴ�������ʱ��arg�Ƕ�`
	Point3 rotate(Point3 p,double ang)
	{
		if(sgn(((s-p)^(e-p)).len()) == 0)return p;
		Point3 f1 = (e-s)^(p-s);
		Point3 f2 = (e-s)^(f1);
		double len = ((s-p)^(e-p)).len()/s.distance(e);
		f1 = f1.trunc(len); f2 = f2.trunc(len);
		Point3 h = p+f2;
		Point3 pp = h+f1;
		return h + ((p-h)*cos(ang)) + ((pp-h)*sin(ang));
	}
	//`����ֱ����`
	bool pointonseg(Point3 p)
	{
		return sgn( ((s-p)^(e-p)).len() ) == 0 && sgn((s-p)*(e-p)) == 0;
	}
};
struct Plane
{
	Point3 a,b,c,o;//`ƽ���ϵ������㣬�Լ�������`
	Plane(){}
	Plane(Point3 _a,Point3 _b,Point3 _c)
	{
		a = _a;
		b = _b;
		c = _c;
		o = pvec();
	}
	Point3 pvec()
	{
		return (b-a)^(c-a);
	}
	//`ax+by+cz+d = 0`
	Plane(double _a,double _b,double _c,double _d)
	{
		o = Point3(_a,_b,_c);
		if(sgn(_a) != 0)
			a = Point3((-_d-_c-_b)/_a,1,1);
		else if(sgn(_b) != 0)
			a = Point3(1,(-_d-_c-_a)/_b,1);
		else if(sgn(_c) != 0)
			a = Point3(1,1,(-_d-_a-_b)/_c);
	}
	//`����ƽ���ϵ��ж�`
	bool pointonplane(Point3 p)
	{
		return sgn((p-a)*o) == 0;
	}
	//`��ƽ��н�`
	double angleplane(Plane f)
	{
		return acos(o*f.o)/(o.len()*f.o.len());
	}
	//`ƽ���ֱ�ߵĽ��㣬����ֵ�ǽ������`
	int crossline(Line3 u,Point3 &p)
	{
		double x = o*(u.e-a);
		double y = o*(u.s-a);
		double d = x-y;
		if(sgn(d) == 0)return 0;
		p = ((u.s*x)-(u.e*y))/d;
		return 1;
	}
	//`�㵽ƽ�������(Ҳ����ͶӰ)`
	Point3 pointtoplane(Point3 p)
	{
		Line3 u = Line3(p,p+o);
		crossline(u,p);
		return p;
	}
	//`ƽ���ƽ��Ľ���`
	int crossplane(Plane f,Line3 &u)
	{
		Point3 oo = o^f.o;
		Point3 v = o^oo;
		double d = fabs(f.o*v);
		if(sgn(d) == 0)return 0;
		Point3 q = a + (v*(f.o*(f.a-a))/d);
		u = Line3(q,q+oo);
		return 1;
	}
};


//���ƽ����
const int MAXN = 100010;
const double eps = 1e-8;
const double INF = 1e20;
struct Point{
	double x,y;
	void input(){
		scanf("%lf%lf",&x,&y);
	}
};
double dist(Point a,Point b){
	return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}
Point p[MAXN];
Point tmpt[MAXN];
bool cmpx(Point a,Point b){
	return a.x < b.x || (a.x == b.x && a.y < b.y);
}
bool cmpy(Point a,Point b){
	return a.y < b.y || (a.y == b.y && a.x < b.x);
}
double Closest_Pair(int left,int right){
	double d = INF;
	if(left == right)return d;
	if(left+1 == right)return dist(p[left],p[right]);
	int mid = (left+right)/2;
	double d1 = Closest_Pair(left,mid);
	double d2 = Closest_Pair(mid+1,right);
	d = min(d1,d2);
	int cnt = 0;
	for(int i = left;i <= right;i++){
		if(fabs(p[mid].x - p[i].x) <= d)
			tmpt[cnt++] = p[i];
	}
	sort(tmpt,tmpt+cnt,cmpy);
	for(int i = 0;i < cnt;i++){
		for(int j = i+1;j < cnt && tmpt[j].y - tmpt[i].y < d;j++)
			d = min(d,dist(tmpt[i],tmpt[j]));
	}
	return d;
}
int main(){
	int n;
	while(scanf("%d",&n) == 1 && n){
		for(int i = 0;i < n;i++)p[i].input();
		sort(p,p+n,cmpx);
		printf("%.2lf\n",Closest_Pair(0,n-1));
	}
    return 0;
}


//��ά͹��  Hud4273
const double eps = 1e-8;
const int MAXN = 550;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0, double _y = 0, double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//���
	double operator *(const Point3 &b)const{
		return x*b.x + y*b.y + z*b.z;
	}
	//���
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
};
struct CH3D{
	struct face{
		//��ʾ͹��һ�����ϵ�������ı��
		int a,b,c;
		//��ʾ�����Ƿ��������յ�͹���ϵ���
		bool ok;
	};
	//��ʼ������
	int n;
	Point3 P[MAXN];
	//͹���������������
	int num;
	//͹�������������
	face F[8*MAXN];
	int g[MAXN][MAXN];
	//���
	Point3 cross(const Point3 &a,const Point3 &b,const Point3 &c){
		return (b-a)^(c-a);
	}
	//`���������*2`
	double area(Point3 a,Point3 b,Point3 c){
		return ((b-a)^(c-a)).len();
	}
	//`�������������*6`
	double volume(Point3 a,Point3 b,Point3 c,Point3 d){
		return ((b-a)^(c-a))*(d-a);
	}
	//`����������ͬ��`
	double dblcmp(Point3 &p,face &f){
		Point3 p1 = P[f.b] - P[f.a];
		Point3 p2 = P[f.c] - P[f.a];
		Point3 p3 = p - P[f.a];
		return (p1^p2)*p3;
	}
	void deal(int p,int a,int b){
		int f = g[a][b];
		face add;
		if(F[f].ok){
			if(dblcmp(P[p],F[f]) > eps)
				dfs(p,f);
			else {
				add.a = b;
				add.b = a;
				add.c = p;
				add.ok = true;
				g[p][b] = g[a][p] = g[b][a] = num;
				F[num++] = add;
			}
		}
	}
	//�ݹ���������Ӧ�ô�͹����ɾ������
	void dfs(int p,int now){
		F[now].ok = false;
		deal(p,F[now].b,F[now].a);
		deal(p,F[now].c,F[now].b);
		deal(p,F[now].a,F[now].c);
	}
	bool same(int s,int t){
		Point3 &a = P[F[s].a];
		Point3 &b = P[F[s].b];
		Point3 &c = P[F[s].c];
		return fabs(volume(a,b,c,P[F[t].a])) < eps &&
			fabs(volume(a,b,c,P[F[t].b])) < eps &&
			fabs(volume(a,b,c,P[F[t].c])) < eps;
	}
	//������ά͹��
	void create(){
		num = 0;
		face add;

		//***********************************
		//�˶���Ϊ�˱�֤ǰ�ĸ��㲻����
		bool flag = true;
		for(int i = 1;i < n;i++){
			if(!(P[0] == P[i])){
				swap(P[1],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		flag = true;
		for(int i = 2;i < n;i++){
			if( ((P[1]-P[0])^(P[i]-P[0])).len() > eps ){
				swap(P[2],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		flag = true;
		for(int i = 3;i < n;i++){
			if(fabs( ((P[1]-P[0])^(P[2]-P[0]))*(P[i]-P[0]) ) > eps){
				swap(P[3],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		//**********************************

		for(int i = 0;i < 4;i++){
			add.a = (i+1)%4;
			add.b = (i+2)%4;
			add.c = (i+3)%4;
			add.ok = true;
			if(dblcmp(P[i],add) > 0)swap(add.b,add.c);
			g[add.a][add.b] = g[add.b][add.c] = g[add.c][add.a] = num;
			F[num++] = add;
		}
		for(int i = 4;i < n;i++)
			for(int j = 0;j < num;j++)
				if(F[j].ok && dblcmp(P[i],F[j]) > eps){
					dfs(i,j);
					break;
				}
		int tmp = num;
		num = 0;
		for(int i = 0;i < tmp;i++)
			if(F[i].ok)
				F[num++] = F[i];
	}
	//�����
	//`���ԣ�HDU3528`
	double area(){
		double res = 0;
		if(n == 3){
			Point3 p = cross(P[0],P[1],P[2]);
			return p.len()/2;
		}
		for(int i = 0;i < num;i++)
			res += area(P[F[i].a],P[F[i].b],P[F[i].c]);
		return res/2.0;
	}
	double volume(){
		double res = 0;
		Point3 tmp = Point3(0,0,0);
		for(int i = 0;i < num;i++)
			res += volume(tmp,P[F[i].a],P[F[i].b],P[F[i].c]);
		return fabs(res/6);
	}
	//���������θ���
	int triangle(){
		return num;
	}
	//�������θ���
	//`���ԣ�HDU3662`
	int polygon(){
		int res = 0;
		for(int i = 0;i < num;i++){
			bool flag = true;
			for(int j = 0;j < i;j++)
				if(same(i,j)){
					flag = 0;
					break;
				}
			res += flag;
		}
		return res;
	}
	//����
	//`���ԣ�HDU4273`
	Point3 barycenter(){
		Point3 ans = Point3(0,0,0);
		Point3 o = Point3(0,0,0);
		double all = 0;
		for(int i = 0;i < num;i++){
			double vol = volume(o,P[F[i].a],P[F[i].b],P[F[i].c]);
			ans = ans + (((o+P[F[i].a]+P[F[i].b]+P[F[i].c])/4.0)*vol);
			all += vol;
		}
		ans = ans/all;
		return ans;
	}
	//�㵽��ľ���
	//`���ԣ�HDU4273`
	double ptoface(Point3 p,int i){
		double tmp1 = fabs(volume(P[F[i].a],P[F[i].b],P[F[i].c],p));
		double tmp2 = ((P[F[i].b]-P[F[i].a])^(P[F[i].c]-P[F[i].a])).len();
		return tmp1/tmp2;
	}
};
CH3D hull;
int main()
{
    while(scanf("%d",&hull.n) == 1){
		for(int i = 0;i < hull.n;i++)hull.P[i].input();
		hull.create();
		Point3 p = hull.barycenter();
		double ans = 1e20;
		for(int i = 0;i < hull.num;i++)
			ans = min(ans,hull.ptoface(p,i));
		printf("%.3lf\n",ans);
	}
    return 0;
}

