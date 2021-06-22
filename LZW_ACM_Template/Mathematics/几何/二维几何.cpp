// ��ά���㼸��ģ��`
const double eps = 1e-8;
const double inf = 1e20;
const double pi = acos(-1.0);
const int maxp = 1010;
//`Compares a double to zero`
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
//square of a double
inline double sqr(double x){return x*x;}
/*
 * Point
 * Point()               - Empty constructor
 * Point(double _x,double _y)  - constructor
 * input()             - double input
 * output()            - %.2f output
 * operator ==         - compares x and y
 * operator <          - compares first by x, then by y
 * operator -          - return new Point after subtracting curresponging x and y
 * operator ^          - cross product of 2d points
 * operator *          - dot product
 * len()               - gives length from origin
 * len2()              - gives square of length from origin
 * distance(Point p)   - gives distance from p
 * operator + Point b  - returns new Point after adding curresponging x and y
 * operator * double k - returns new Point after multiplieing x and y by k
 * operator / double k - returns new Point after divideing x and y by k
 * rad(Point a,Point b)- returns the angle of Point a and Point b from this Point
 * trunc(double r)     - return Point that if truncated the distance from center to r
 * rotleft()           - returns 90 degree ccw rotated point
 * rotright()          - returns 90 degree cw rotated point
 * rotate(Point p,double angle) - returns Point after rotateing the Point centering at p by angle radian ccw
 */
struct Point{               //��/����
	double x,y;
	Point(){}
	Point(double _x,double _y){
		x = _x;
		y = _y;
	}
	void input(){
		scanf("%lf%lf",&x,&y);
	}
	void output(){
		printf("%.2f %.2f\n",x,y);
	}
	bool operator == (Point b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0;
	}
	bool operator < (Point b)const{                //���ȱȽ�x�Ĵ�С��x��ͬʱ�Ƚ�y�����ڵ�����
		return sgn(x-b.x)== 0?sgn(y-b.y)<0:x<b.x;
	}
	Point operator -(const Point &b)const{
		return Point(x-b.x,y-b.y);
	}
	//���
	double operator ^(const Point &b)const{
		return x*b.y - y*b.x;
	}
	//���
	double operator *(const Point &b)const{
		return x*b.x + y*b.y;
	}
	//���س��� �㵽ԭ��ľ���
	double len(){
		return hypot(x,y);//�⺯��
	}
	//���س��ȵ�ƽ��
	double len2(){
		return x*x + y*y;
	}
	//��������ľ���
	double distance(Point p){
		return hypot(x-p.x,y-p.y);
	}
	Point operator +(const Point &b)const{
		return Point(x+b.x,y+b.y);
	}
	Point operator *(const double &k)const{
		return Point(x*k,y*k);
	}
	Point operator /(const double &k)const{
		return Point(x/k,y/k);
	}
	//`����pa  ��  pb �ļн�`
	//`����������㿴a,b ���ɵļн�`
	//`���� LightOJ1203`
	double rad(Point a,Point b){
		Point p = *this;
		return fabs(atan2( fabs((a-p)^(b-p)),(a-p)*(b-p) ));
	}
	//`��Ϊ����Ϊr������`
	Point trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point(x*r,y*r);
	}
	//`��ʱ����ת90��`
	Point rotleft(){
		return Point(-y,x);
	}
	//`˳ʱ����ת90��`
	Point rotright(){
		return Point(y,-x);
	}
	//`����p����ʱ����תangle`
	Point rotate(Point p,double angle){
		Point v = (*this) - p;
		double c = cos(angle), s = sin(angle);
		return Point(p.x + v.x*c - v.y*s,p.y + v.x*s + v.y*c);
	}
};



/*
 * Stores two points
 * Line()                         - Empty constructor
 * Line(Point _s,Point _e)        - Line through _s and _e
 * operator ==                    - checks if two points are same
 * Line(Point p,double angle)     - one end p , another end at angle degree
 * Line(double a,double b,double c) - Line of equation ax + by + c = 0
 * input()                        - inputs s and e
 * adjust()                       - orders in such a way that s < e
 * length()                       - distance of se
 * angle()                        - return 0 <= angle < pi
 * relation(Point p)              - 3 if point is on line
 *                                  1 if point on the left of line
 *                                  2 if point on the right of line
 * pointonseg(double p)           - return true if point on segment
 * parallel(Line v)               - return true if they are parallel
 * segcrossseg(Line v)            - returns 0 if does not intersect
 *                                  returns 1 if non-standard intersection
 *                                  returns 2 if intersects
 * linecrossseg(Line v)           - line and seg
 * linecrossline(Line v)          - 0 if parallel
 *                                  1 if coincides
 *                                  2 if intersects
 * crosspoint(Line v)             - returns intersection point
 * dispointtoline(Point p)        - distance from point p to the line
 * dispointtoseg(Point p)         - distance from p to the segment
 * dissegtoseg(Line v)            - distance of two segment
 * lineprog(Point p)              - returns projected point p on se line
 * symmetrypoint(Point p)         - returns reflection point of p over se
 *
 */
//�ṹ�� ֱ��/�߶�
struct Line{
	Point s,e;
	Line(){}
	Line(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	bool operator ==(Line v){
		return (s == v.s)&&(e == v.e);
	}
	//`����һ�������б��angleȷ��ֱ��,0<=angle<pi`
	Line(Point p,double angle){
		s = p;
		if(sgn(angle-pi/2) == 0){
			e = (s + Point(0,1));
		}
		else{
			e = (s + Point(1,tan(angle)));
		}
	}
	//ax+by+c=0
	Line(double a,double b,double c){
		if(sgn(a) == 0){
			s = Point(0,-c/b);
			e = Point(1,-c/b);
		}
		else if(sgn(b) == 0){
			s = Point(-c/a,0);
			e = Point(-c/a,1);
		}
		else{
			s = Point(0,-c/b);
			e = Point(1,(-c-a)/b);
		}
	}
	void input(){
		s.input();
		e.input();
	}
	void adjust(){            //����
		if(e < s)swap(s,e);
	}
	//���߶γ���
	double length(){
		return s.distance(e);
	}
	//`����ֱ����б�� 0<=angle<pi`
	double angle(){
		double k = atan2(e.y-s.y,e.x-s.x);
		if(sgn(k) < 0)k += pi;
		if(sgn(k-pi) == 0)k -= pi;
		return k;
	}
	//`���ֱ�߹�ϵ`
	//`1  �����`
	//`2  ���Ҳ�`
	//`3  ��ֱ����`
	int relation(Point p){
		int c = sgn((p-s)^(e-s));
		if(c < 0)return 1;
		else if(c > 0)return 2;
		else return 3;
	}
	// �����߶��ϵ��ж�
	bool pointonseg(Point p){
		return sgn((p-s)^(e-s)) == 0 && sgn((p-s)*(p-e)) <= 0;
	}
	//`������ƽ��(��Ӧֱ��ƽ�л��غ�)`
	bool parallel(Line v){
		return sgn((e-s)^(v.e-v.s)) == 0;
	}
	//`���߶��ཻ�ж�`
	//`2 �淶�ཻ`�����߶εĽ��㲻�����߶ζ˵������غϵ��߶Σ�
	//`1 �ǹ淶�ཻ`
	//`0 ���ཻ`
	int segcrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		int d3 = sgn((v.e-v.s)^(s-v.s));
		int d4 = sgn((v.e-v.s)^(e-v.s));
		if( (d1^d2)==-2 && (d3^d4)==-2 )return 2;
		return (d1==0 && sgn((v.s-s)*(v.s-e))<=0) ||
			(d2==0 && sgn((v.e-s)*(v.e-e))<=0) ||
			(d3==0 && sgn((s-v.s)*(s-v.e))<=0) ||
			(d4==0 && sgn((e-v.s)*(e-v.e))<=0);
	}
	//`ֱ�ߺ��߶��ཻ�ж�`
	//`-*this line   -v seg`
	//`2 �淶�ཻ`
	//`1 �ǹ淶�ཻ`
	//`0 ���ཻ`
	int linecrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		if((d1^d2)==-2) return 2;
		return (d1==0||d2==0);
	}
	//`��ֱ�߹�ϵ`
	//`0 ƽ��`
	//`1 �غ�`
	//`2 �ཻ`
	int linecrossline(Line v){
		if((*this).parallel(v))
			return v.relation(s)==3;
		return 2;
	}
	//`����ֱ�ߵĽ���`
	//`Ҫ��֤��ֱ�߲�ƽ�л��غ�`
	Point crosspoint(Line v){
		double a1 = (v.e-v.s)^(s-v.s);
		double a2 = (v.e-v.s)^(e-v.s);
		return Point((s.x*a2-e.x*a1)/(a2-a1),(s.y*a2-e.y*a1)/(a2-a1));
	}
	//�㵽ֱ�ߵľ���
	double dispointtoline(Point p){
		return fabs((p-s)^(e-s))/length();
	}
	//�㵽�߶εľ���
	double dispointtoseg(Point p){
		if(sgn((p-s)*(e-s))<0 || sgn((p-e)*(s-e))<0)
			return min(p.distance(s),p.distance(e));
		return dispointtoline(p);
	}
	//`�����߶ε��߶εľ���`
	//`ǰ�������߶β��ཻ���ཻ�������0��`
	double dissegtoseg(Line v){
		return min(min(dispointtoseg(v.s),dispointtoseg(v.e)),min(v.dispointtoseg(s),v.dispointtoseg(e)));
	}
	//`���ص�p��ֱ���ϵ�ͶӰ`
	Point lineprog(Point p){
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`���ص�p����ֱ�ߵĶԳƵ�`
	Point symmetrypoint(Point p){
		Point q = lineprog(p);
		return Point(2*q.x-p.x,2*q.y-p.y);
	}
};



//Բ
struct circle{
	Point p;//Բ��
	double r;//�뾶
	circle(){}
	circle(Point _p,double _r){
		p = _p;
		r = _r;
	}
	circle(double x,double y,double _r){
		p = Point(x,y);
		r = _r;
	}
	//`�����ε����Բ`
	//`��ҪPoint��+ /  rotate()  �Լ�Line��crosspoint()`
	//`���������ߵ��д��ߵõ�Բ��`
	//`���ԣ�UVA12304`
	circle(Point a,Point b,Point c){
		Line u = Line((a+b)/2,((a+b)/2)+((b-a).rotleft()));
		Line v = Line((b+c)/2,((b+c)/2)+((c-b).rotleft()));
		p = u.crosspoint(v);
		r = p.distance(a);
	}
	//`�����ε�����Բ`
	//`����bool tû�����ã�ֻ��Ϊ�˺��������Բ��������`
	//`���ԣ�UVA12304`
	circle(Point a,Point b,Point c,bool t){
		Line u,v;
		double m = atan2(b.y-a.y,b.x-a.x), n = atan2(c.y-a.y,c.x-a.x);
		u.s = a;
		u.e = u.s + Point(cos((n+m)/2),sin((n+m)/2));
		v.s = b;
		m = atan2(a.y-b.y,a.x-b.x) , n = atan2(c.y-b.y,c.x-b.x);
		v.e = v.s + Point(cos((n+m)/2),sin((n+m)/2));
		p = u.crosspoint(v);
		r = Line(a,b).dispointtoseg(p);
	}
	//����
	void input(){
		p.input();
		scanf("%lf",&r);
	}
	//���
	void output(){
		printf("%.2lf %.2lf %.2lf\n",p.x,p.y,r);
	}
	bool operator == (circle v){
		return (p==v.p) && sgn(r-v.r)==0;
	}
	bool operator < (circle v)const{
		return ((p<v.p)||((p==v.p)&&sgn(r-v.r)<0));
	}
	//���
	double area(){
		return pi*r*r;
	}
	//�ܳ�
	double circumference(){
		return 2*pi*r;
	}
	//`���Բ�Ĺ�ϵ`
	//`0 Բ��`
	//`1 Բ��`
	//`2 Բ��`
	int relation(Point b){
		double dst = b.distance(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r)==0)return 1;
		return 0;
	}
	//`�߶κ�Բ�Ĺ�ϵ`
	//`�Ƚϵ���Բ�ĵ��߶εľ���Ͱ뾶�Ĺ�ϵ`
	int relationseg(Line v){
		double dst = v.dispointtoseg(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`ֱ�ߺ�Բ�Ĺ�ϵ`
	//`�Ƚϵ���Բ�ĵ�ֱ�ߵľ���Ͱ뾶�Ĺ�ϵ`
	int relationline(Line v){
		double dst = v.dispointtoline(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`��Բ�Ĺ�ϵ`
	//`5 ����`
	//`4 ����`
	//`3 �ཻ`
	//`2 ����`
	//`1 �ں�`
	//`��ҪPoint��distance`
	//`���ԣ�UVA12304`
	int relationcircle(circle v){
		double d = p.distance(v.p);
		if(sgn(d-r-v.r) > 0)return 5;
		if(sgn(d-r-v.r) == 0)return 4;
		double l = fabs(r-v.r);
		if(sgn(d-r-v.r)<0 && sgn(d-l)>0)return 3;
		if(sgn(d-l)==0)return 2;
		if(sgn(d-l)<0)return 1;
	}
	//`������Բ�Ľ��㣬����0��ʾû�н��㣬����1��һ�����㣬2����������`
	//`��Ҫrelationcircle`
	//`���ԣ�UVA12304`
	int pointcrosscircle(circle v,Point &p1,Point &p2){
		int rel = relationcircle(v);
		if(rel == 1 || rel == 5)return 0;
		double d = p.distance(v.p);
		double l = (d*d+r*r-v.r*v.r)/(2*d);
		double h = sqrt(r*r-l*l);
		Point tmp = p + (v.p-p).trunc(l);
		p1 = tmp + ((v.p-p).rotleft().trunc(h));
		p2 = tmp + ((v.p-p).rotright().trunc(h));
		if(rel == 2 || rel == 4)
			return 1;
		return 2;
	}
	//`��ֱ�ߺ�Բ�Ľ��㣬���ؽ������`
	int pointcrossline(Line v,Point &p1,Point &p2){
		if(!(*this).relationline(v))return 0;
		Point a = v.lineprog(p);
		double d = v.dispointtoline(p);
		d = sqrt(r*r-d*d);
		if(sgn(d) == 0){
			p1 = a;
			p2 = a;
			return 1;
		}
		p1 = a + (v.e-v.s).trunc(d);
		p2 = a - (v.e-v.s).trunc(d);
		return 2;
	}
	//`�õ���a,b���㣬�뾶Ϊr1������Բ`
	int gercircle(Point a,Point b,double r1,circle &c1,circle &c2){
		circle x(a,r1),y(b,r1);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r;
		return t;
	}
	//`�õ���ֱ��u���У�����q,�뾶Ϊr1��Բ`
	//`���ԣ�UVA12304`
	int getcircle(Line u,Point q,double r1,circle &c1,circle &c2){
		double dis = u.dispointtoline(q);
		if(sgn(dis-r1*2)>0)return 0;
		if(sgn(dis) == 0){
			c1.p = q + ((u.e-u.s).rotleft().trunc(r1));
			c2.p = q + ((u.e-u.s).rotright().trunc(r1));
			c1.r = c2.r = r1;
			return 2;
		}
		Line u1 = Line((u.s + (u.e-u.s).rotleft().trunc(r1)),(u.e + (u.e-u.s).rotleft().trunc(r1)));
		Line u2 = Line((u.s + (u.e-u.s).rotright().trunc(r1)),(u.e + (u.e-u.s).rotright().trunc(r1)));
		circle cc = circle(q,r1);
		Point p1,p2;
		if(!cc.pointcrossline(u1,p1,p2))cc.pointcrossline(u2,p1,p2);
		c1 = circle(p1,r1);
		if(p1 == p2){
			c2 = c1;
			return 1;
		}
		c2 = circle(p2,r1);
		return 2;
	}
	//`ͬʱ��ֱ��u,v���У��뾶Ϊr1��Բ`
	//`���ԣ�UVA12304`
	int getcircle(Line u,Line v,double r1,circle &c1,circle &c2,circle &c3,circle &c4){
		if(u.parallel(v))return 0;//��ֱ��ƽ��
		Line u1 = Line(u.s + (u.e-u.s).rotleft().trunc(r1),u.e + (u.e-u.s).rotleft().trunc(r1));
		Line u2 = Line(u.s + (u.e-u.s).rotright().trunc(r1),u.e + (u.e-u.s).rotright().trunc(r1));
		Line v1 = Line(v.s + (v.e-v.s).rotleft().trunc(r1),v.e + (v.e-v.s).rotleft().trunc(r1));
		Line v2 = Line(v.s + (v.e-v.s).rotright().trunc(r1),v.e + (v.e-v.s).rotright().trunc(r1));
		c1.r = c2.r = c3.r = c4.r = r1;
		c1.p = u1.crosspoint(v1);
		c2.p = u1.crosspoint(v2);
		c3.p = u2.crosspoint(v1);
		c4.p = u2.crosspoint(v2);
		return 4;
	}
	//`ͬʱ�벻�ཻԲcx,cy���У��뾶Ϊr1��Բ`
	//`���ԣ�UVA12304`
	int getcircle(circle cx,circle cy,double r1,circle &c1,circle &c2){
		circle x(cx.p,r1+cx.r),y(cy.p,r1+cy.r);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r1;
		return t;
	}

	//`��һ����Բ������(���жϵ��Բ�Ĺ�ϵ)`
	//`���ԣ�UVA12304`
	int tangentline(Point q,Line &u,Line &v){
		int x = relation(q);
		if(x == 2)return 0;
		if(x == 1){
			u = Line(q,q + (q-p).rotleft());
			v = u;
			return 1;
		}
		double d = p.distance(q);
		double l = r*r/d;
		double h = sqrt(r*r-l*l);
		u = Line(q,p + ((q-p).trunc(l) + (q-p).rotleft().trunc(h)));
		v = Line(q,p + ((q-p).trunc(l) + (q-p).rotright().trunc(h)));
		return 2;
	}
	//`����Բ�ཻ�����`
	double areacircle(circle v){
		int rel = relationcircle(v);
		if(rel >= 4)return 0.0;
		if(rel <= 2)return min(area(),v.area());
		double d = p.distance(v.p);
		double hf = (r+v.r+d)/2.0;
		double ss = 2*sqrt(hf*(hf-r)*(hf-v.r)*(hf-d));
		double a1 = acos((r*r+d*d-v.r*v.r)/(2.0*r*d));
		a1 = a1*r*r;
		double a2 = acos((v.r*v.r+d*d-r*r)/(2.0*v.r*d));
		a2 = a2*v.r*v.r;
		return a1+a2-ss;
	}
	//`��Բ��������pab���ཻ���`
	//`���ԣ�POJ3675 HDU3982 HDU2892`
	double areatriangle(Point a,Point b){
		if(sgn((p-a)^(p-b)) == 0)return 0.0;
		Point q[5];
		int len = 0;
		q[len++] = a;
		Line l(a,b);
		Point p1,p2;
		if(pointcrossline(l,q[1],q[2])==2){
			if(sgn((a-q[1])*(b-q[1]))<0)q[len++] = q[1];
			if(sgn((a-q[2])*(b-q[2]))<0)q[len++] = q[2];
		}
		q[len++] = b;
		if(len == 4 && sgn((q[0]-q[1])*(q[2]-q[1]))>0)swap(q[1],q[2]);
		double res = 0;
		for(int i = 0;i < len-1;i++){
			if(relation(q[i])==0||relation(q[i+1])==0){
				double arg = p.rad(q[i],q[i+1]);
				res += r*r*arg/2.0;
			}
			else{
				res += fabs((q[i]-p)^(q[i+1]-p))/2.0;
			}
		}
		return res;
	}
};



/*
 * n,p  Line l for each side
 * input(int _n)                        - inputs _n size polygon
 * add(Point q)                         - adds a point at end of the list
 * getline()                            - populates line array
 * cmp                                  - comparision in convex_hull order
 * norm()                               - sorting in convex_hull order
 * getconvex(polygon &convex)           - returns convex hull in convex
 * Graham(polygon &convex)              - returns convex hull in convex
 * isconvex()                           - checks if convex
 * relationpoint(Point q)               - returns 3 if q is a vertex
 *                                                2 if on a side
 *                                                1 if inside
 *                                                0 if outside
 * convexcut(Line u,polygon &po)        - left side of u in po
 * gercircumference()                   - returns side length
 * getarea()                            - returns area
 * getdir()                             - returns 0 for cw, 1 for ccw
 * getbarycentre()                      - returns barycenter
 *
 */




//�����
struct polygon{
	int n;
	Point p[maxp];
	Line l[maxp];
	void input(int _n){
		n = _n;
		for(int i = 0;i < n;i++)
			p[i].input();
	}
	void add(Point q){
		p[n++] = q;
	}
	void getline(){
		for(int i = 0;i < n;i++){
			l[i] = Line(p[i],p[(i+1)%n]);
		}
	}
	struct cmp{
		Point p;
		cmp(const Point &p0){p = p0;}
		bool operator()(const Point &aa,const Point &bb){
			Point a = aa, b = bb;
			int d = sgn((a-p)^(b-p));
			if(d == 0){
				return sgn(a.distance(p)-b.distance(p)) < 0;
			}
			return d > 0;
		}
	};
	//`���м�������`
	//`������Ҫ�ҵ������½ǵĵ�`
	//`��Ҫ���غź�Point�� < ������(min����Ҫ��) `
	void norm(){
		Point mi = p[0];
		for(int i = 1;i < n;i++)mi = min(mi,p[i]);
		sort(p,p+n,cmp(mi));
	}
	//`�õ�͹��`
	//`�õ���͹������ĵ�����0~n-1��`
	//`����͹���ķ���`
	//`ע�������Ӱ�죬Ҫ���������е㹲�㣬���߹��ߵ��������`
	//`���� LightOJ1203  LightOJ1239`
	void getconvex(polygon &convex){
		sort(p,p+n);
		convex.n = n;
		for(int i = 0;i < min(n,2);i++){
			convex.p[i] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//����
		if(n <= 2)return;
		int &top = convex.n;
		top = 1;
		for(int i = 2;i < n;i++){
			while(top && sgn((convex.p[top]-p[i])^(convex.p[top-1]-p[i])) <= 0)
				top--;
			convex.p[++top] = p[i];
		}
		int temp = top;
		convex.p[++top] = p[n-2];
		for(int i = n-3;i >= 0;i--){
			while(top != temp && sgn((convex.p[top]-p[i])^(convex.p[top-1]-p[i])) <= 0)
				top--;
			convex.p[++top] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//����
		convex.norm();//`ԭ���õ�����˳ʱ��ĵ㣬�������ʱ��`
	}
	//`�õ�͹��������һ�ַ���`
	//`���� LightOJ1203  LightOJ1239`
	void Graham(polygon &convex){
		norm();
		int &top = convex.n;
		top = 0;
		if(n == 1){
			top = 1;
			convex.p[0] = p[0];
			return;
		}
		if(n == 2){
			top = 2;
			convex.p[0] = p[0];
			convex.p[1] = p[1];
			if(convex.p[0] == convex.p[1])top--;
			return;
		}
		convex.p[0] = p[0];
		convex.p[1] = p[1];
		top = 2;
		for(int i = 2;i < n;i++){
			while( top > 1 && sgn((convex.p[top-1]-convex.p[top-2])^(p[i]-convex.p[top-2])) <= 0 )
				top--;
			convex.p[top++] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//����
	}
	//`�ж��ǲ���͹��`
	bool isconvex(){
		bool s[2];
		memset(s,false,sizeof(s));
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			int k = (j+1)%n;
			s[sgn((p[j]-p[i])^(p[k]-p[i]))+1] = true;
			if(s[0] && s[2])return false;
		}
		return true;
	}
	//`�жϵ���������εĹ�ϵ`
	//` 3 ����`
	//` 2 ����`
	//` 1 �ڲ�`
	//` 0 �ⲿ`
	int relationpoint(Point q){
		for(int i = 0;i < n;i++){
			if(p[i] == q)return 3;
		}
		getline();
		for(int i = 0;i < n;i++){
			if(l[i].pointonseg(q))return 2;
		}
		int cnt = 0;
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			int k = sgn((q-p[j])^(p[i]-p[j]));
			int u = sgn(p[i].y-q.y);
			int v = sgn(p[j].y-q.y);
			if(k > 0 && u < 0 && v >= 0)cnt++;
			if(k < 0 && v < 0 && u >= 0)cnt--;
		}
		return cnt != 0;
	}
	//`ֱ��u�и�͹��������`
	//`ע��ֱ�߷���`
	//`���ԣ�HDU3982`
	void convexcut(Line u,polygon &po){
		int &top = po.n;//ע������
		top = 0;
		for(int i = 0;i < n;i++){
			int d1 = sgn((u.e-u.s)^(p[i]-u.s));
			int d2 = sgn((u.e-u.s)^(p[(i+1)%n]-u.s));
			if(d1 >= 0)po.p[top++] = p[i];
			if(d1*d2 < 0)po.p[top++] = u.crosspoint(Line(p[i],p[(i+1)%n]));
		}
	}
	//`�õ��ܳ�`
	//`���� LightOJ1239`
	double getcircumference(){
		double sum = 0;
		for(int i = 0;i < n;i++){
			sum += p[i].distance(p[(i+1)%n]);
		}
		return sum;
	}
	//`�õ����`
	double getarea(){
		double sum = 0;
		for(int i = 0;i < n;i++){
			sum += (p[i]^p[(i+1)%n]);
		}
		return fabs(sum)/2;
	}
	//`�õ�����`
	//` 1 ��ʾ��ʱ�룬0��ʾ˳ʱ��`
	bool getdir(){
		double sum = 0;
		for(int i = 0;i < n;i++)
			sum += (p[i]^p[(i+1)%n]);
		if(sgn(sum) > 0)return 1;
		return 0;
	}
	//`�õ�����`
	Point getbarycentre(){
		Point ret(0,0);
		double area = 0;
		for(int i = 1;i < n-1;i++){
			double tmp = (p[i]-p[0])^(p[i+1]-p[0]);
			if(sgn(tmp) == 0)continue;
			area += tmp;
			ret.x += (p[0].x+p[i].x+p[i+1].x)/3*tmp;
			ret.y += (p[0].y+p[i].y+p[i+1].y)/3*tmp;
		}
		if(sgn(area)) ret = ret/area;
		return ret;
	}
	//`����κ�Բ�������`
	//`���ԣ�POJ3675 HDU3982 HDU2892`
	double areacircle(circle c){
		double ans = 0;
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			if(sgn( (p[j]-c.p)^(p[i]-c.p) ) >= 0)
				ans += c.areatriangle(p[i],p[j]);
			else ans -= c.areatriangle(p[i],p[j]);
		}
		return fabs(ans);
	}
	//`����κ�Բ��ϵ`
	//` 2 Բ��ȫ�ڶ������`
	//` 1 Բ�ڶ�������棬�����˶���α߽�`
	//` 0 ����`
	int relationcircle(circle c){
		getline();
		int x = 2;
		if(relationpoint(c.p) != 1)return 0;//Բ�Ĳ����ڲ�
		for(int i = 0;i < n;i++){
			if(c.relationseg(l[i])==2)return 0;
			if(c.relationseg(l[i])==1)x = 1;
		}
		return x;
	}
};
//`AB X AC`
double cross(Point A,Point B,Point C){
	return (B-A)^(C-A);
}
//`AB*AC`
double dot(Point A,Point B,Point C){
	return (B-A)*(C-A);
}
//`��С�����������`
//` A ������͹��(��������ʱ��˳��)`
//` ���� UVA 10173`
double minRectangleCover(polygon A){
	//`Ҫ����A.n < 3�����`
	if(A.n < 3)return 0.0;
	A.p[A.n] = A.p[0];
	double ans = -1;
	int r = 1, p = 1, q;
	for(int i = 0;i < A.n;i++){
		//`�������A.p[i] - A.p[i+1]��Զ�ĵ�`
		while( sgn( cross(A.p[i],A.p[i+1],A.p[r+1]) - cross(A.p[i],A.p[i+1],A.p[r]) ) >= 0 )
			r = (r+1)%A.n;
		//`����A.p[i] - A.p[i+1]����������n��Զ�ĵ�`
		while(sgn( dot(A.p[i],A.p[i+1],A.p[p+1]) - dot(A.p[i],A.p[i+1],A.p[p]) ) >= 0 )
			p = (p+1)%A.n;
		if(i == 0)q = p;
		//`����A.p[i] - A.p[i+1]�����ϸ�����Զ�ĵ�`
		while(sgn(dot(A.p[i],A.p[i+1],A.p[q+1]) - dot(A.p[i],A.p[i+1],A.p[q])) <= 0)
			q = (q+1)%A.n;
		double d = (A.p[i] - A.p[i+1]).len2();
		double tmp = cross(A.p[i],A.p[i+1],A.p[r]) *
			(dot(A.p[i],A.p[i+1],A.p[p]) - dot(A.p[i],A.p[i+1],A.p[q]))/d;
		if(ans < 0 || ans > tmp)ans = tmp;
	}
	return ans;
}

//`ֱ����͹�����`
//`���������ʱ��ģ���q1q2�����`
//`����:HDU3982`
vector<Point> convexCut(const vector<Point> &ps,Point q1,Point q2){
	vector<Point>qs;
	int n = ps.size();
	for(int i = 0;i < n;i++){
		Point p1 = ps[i], p2 = ps[(i+1)%n];
		int d1 = sgn((q2-q1)^(p1-q1)), d2 = sgn((q2-q1)^(p2-q1));
		if(d1 >= 0)
			qs.push_back(p1);
		if(d1 * d2 < 0)
			qs.push_back(Line(p1,p2).crosspoint(Line(q1,q2)));
	}
	return qs;
}




//`��ƽ�潻`
//`���� POJ3335 POJ1474 POJ1279`
//***************************
struct halfplane:public Line{
	double angle;
	halfplane(){}
	//`��ʾ����s->e��ʱ��(���)�İ�ƽ��`
	halfplane(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	halfplane(Line v){
		s = v.s;
		e = v.e;
	}
	void calcangle(){
		angle = atan2(e.y-s.y,e.x-s.x);
	}
	bool operator <(const halfplane &b)const{
		return angle < b.angle;
	}
};
struct halfplanes{
	int n;
	halfplane hp[maxp];
	Point p[maxp];
	int que[maxp];
	int st,ed;
	halfplanes(){n=0;}
	void push(halfplane tmp){
		hp[n++] = tmp;
	}
	//ȥ��
	void unique(){
		int m = 1;
		for(int i = 1;i < n;i++){
			if(sgn(hp[i].angle-hp[i-1].angle) != 0)
				hp[m++] = hp[i];
			else if(sgn( (hp[m-1].e-hp[m-1].s)^(hp[i].s-hp[m-1].s) ) > 0)
				hp[m-1] = hp[i];
		}
		n = m;
	}
	bool halfplaneinsert(){
		for(int i = 0;i < n;i++)hp[i].calcangle();
		sort(hp,hp+n);
		unique();
		que[st=0] = 0;
		que[ed=1] = 1;
		p[1] = hp[0].crosspoint(hp[1]);
		for(int i = 2;i < n;i++){
			while(st<ed && sgn((hp[i].e-hp[i].s)^(p[ed]-hp[i].s))<0)ed--;
			while(st<ed && sgn((hp[i].e-hp[i].s)^(p[st+1]-hp[i].s))<0)st++;
			que[++ed] = i;
			if(hp[i].parallel(hp[que[ed-1]]))return false;
			p[ed]=hp[i].crosspoint(hp[que[ed-1]]);
		}
		while(st<ed && sgn((hp[que[st]].e-hp[que[st]].s)^(p[ed]-hp[que[st]].s))<0)ed--;
		while(st<ed && sgn((hp[que[ed]].e-hp[que[ed]].s)^(p[st+1]-hp[que[ed]].s))<0)st++;
		if(st+1>=ed)return false;
		return true;
	}
	//`�õ�����ƽ�潻�õ���͹�����`
	//`��Ҫ�ȵ���halfplaneinsert() �ҷ���true`
	void getconvex(polygon &con){
		p[st] = hp[que[st]].crosspoint(hp[que[ed]]);
		con.n = ed-st+1;
		for(int j = st,i = 0;j <= ed;i++,j++)
			con.p[i] = p[j];
	}
};
//***************************









//��Բ/Բ��

const int maxn = 1010;
struct circles{
	circle c[maxn];
	double ans[maxn];//`ans[i]��ʾ��������i�ε����`
	double pre[maxn];
	int n;
	circles(){}
	void add(circle cc){
		c[n++] = cc;
	}
	//`x������y��`
	bool inner(circle x,circle y){
		if(x.relationcircle(y) != 1)return 0;
		return sgn(x.r-y.r)<=0?1:0;
	}
	//Բ�������ȥ���ں���Բ
	void init_or(){
		bool mark[maxn] = {0};
		int i,j,k=0;
		for(i = 0;i < n;i++){
			for(j = 0;j < n;j++)
				if(i != j && !mark[j]){
					if( (c[i]==c[j])||inner(c[i],c[j]) )break;
				}
			if(j < n)mark[i] = 1;
		}
		for(i = 0;i < n;i++)
			if(!mark[i])
				c[k++] = c[i];
		n = k;
	}
	//`Բ�������ȥ���ں���Բ`
	void init_add(){
		int i,j,k;
		bool mark[maxn] = {0};
		for(i = 0;i < n;i++){
			for(j = 0;j < n;j++)
				if(i != j && !mark[j]){
					if( (c[i]==c[j])||inner(c[j],c[i]) )break;
				}
			if(j < n)mark[i] = 1;
		}
		for(i = 0;i < n;i++)
			if(!mark[i])
				c[k++] = c[i];
		n = k;
	}
	//`�뾶Ϊr��Բ������Ϊth��Ӧ�Ĺ��ε����`
	double areaarc(double th,double r){
		return 0.5*r*r*(th-sin(th));
	}
	//`����SPOJVCIRCLES SPOJCIRUT`
	//`SPOJVCIRCLES��n��Բ�����������Ҫ����init\_or()ȥ���ظ�Բ������WA��`
	//`SPOJCIRUT ���󱻸���k�ε���������ܼ�init\_or()`
	//`�����󸲸Ƕ��ٴ���������⣬���ܽ����ͬԲ�����Ҳ���init\_or()`
	//`���Բ���������Ҫinit\_or,����һ��Ŀ�ľ���ȥ����ͬԲ`
	void getarea(){
		memset(ans,0,sizeof(ans));
		vector<pair<double,int> >v;
		for(int i = 0;i < n;i++){
			v.clear();
			v.push_back(make_pair(-pi,1));
			v.push_back(make_pair(pi,-1));
			for(int j = 0;j < n;j++)
				if(i != j){
					Point q = (c[j].p - c[i].p);
					double ab = q.len(),ac = c[i].r, bc = c[j].r;
					if(sgn(ab+ac-bc)<=0){
						v.push_back(make_pair(-pi,1));
						v.push_back(make_pair(pi,-1));
						continue;
					}
					if(sgn(ab+bc-ac)<=0)continue;
					if(sgn(ab-ac-bc)>0)continue;
					double th = atan2(q.y,q.x), fai = acos((ac*ac+ab*ab-bc*bc)/(2.0*ac*ab));
					double a0 = th-fai;
					if(sgn(a0+pi)<0)a0+=2*pi;
					double a1 = th+fai;
					if(sgn(a1-pi)>0)a1-=2*pi;
					if(sgn(a0-a1)>0){
						v.push_back(make_pair(a0,1));
						v.push_back(make_pair(pi,-1));
						v.push_back(make_pair(-pi,1));
						v.push_back(make_pair(a1,-1));
					}
					else{
						v.push_back(make_pair(a0,1));
						v.push_back(make_pair(a1,-1));
					}
				}
			sort(v.begin(),v.end());
			int cur = 0;
			for(int j = 0;j < v.size();j++){
				if(cur && sgn(v[j].first-pre[cur])){
					ans[cur] += areaarc(v[j].first-pre[cur],c[i].r);
					ans[cur] += 0.5*(Point(c[i].p.x+c[i].r*cos(pre[cur]),c[i].p.y+c[i].r*sin(pre[cur]))^Point(c[i].p.x+c[i].r*cos(v[j].first),c[i].p.y+c[i].r*sin(v[j].first)));
				}
				cur += v[j].second;
				pre[cur] = v[j].first;
			}
		}
		for(int i = 1;i < n;i++)
			ans[i] -= ans[i+1];
	}
};