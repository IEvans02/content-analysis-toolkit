import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AIPersona:
    """Represents an AI decision-maker persona"""
    
    def __init__(self, persona_data):
        self.name = persona_data['name']
        self.role = persona_data['role']
        self.company_context = persona_data['company_context']
        self.decision_style = persona_data['decision_style']
        self.personality_bias = persona_data['personality_bias']
        self.pressure_points = persona_data['pressure_points']
        self.communication_style = persona_data['communication_style']
        self.priorities = persona_data['priorities']
        self.common_objections = persona_data['common_objections']
        
    def get_system_prompt(self):
        """Generate the system prompt for this persona"""
        return f"""You are {self.name}, a {self.role} at a {self.company_context['size']} {self.company_context['industry']} company.

PERSONALITY & STYLE:
- Decision Style: {self.decision_style}
- Personality: {self.personality_bias}
- Communication: {self.communication_style}

CONTEXT & PRESSURES:
- Company Size: {self.company_context['size']}
- Industry: {self.company_context['industry']}
- Current Challenges: {', '.join(self.company_context['challenges'])}
- Key Pressure Points: {', '.join(self.pressure_points)}

YOUR PRIORITIES (in order):
{chr(10).join([f"{i+1}. {priority}" for i, priority in enumerate(self.priorities)])}

TYPICAL CONCERNS:
{chr(10).join([f"- {objection}" for objection in self.common_objections])}

When evaluating messaging, you should respond authentically as this persona. Consider:
- How this message addresses your priorities
- Whether it acknowledges your pressure points
- If the tone and style resonate with you
- What concerns or objections you might have
- How credible and differentiated the message feels to you

Provide specific, actionable feedback that reflects your role and personality."""

class PersonaTestingSystem:
    """Main system for AI persona message testing"""
    
    def __init__(self):
        self.personas = self._load_default_personas()
        
    def _load_default_personas(self):
        """Load the default persona library"""
        
        # CIO Personas
        cio_personas = {
            "risk_averse_cio": {
                "name": "Sarah Chen",
                "role": "Chief Information Officer",
                "company_context": {
                    "size": "large enterprise",
                    "industry": "financial services",
                    "challenges": ["regulatory compliance", "cybersecurity threats", "legacy system modernization"]
                },
                "decision_style": "Consensus-building, thorough evaluation, prefers proven solutions",
                "personality_bias": "Risk-averse, values stability, prefers established vendors",
                "pressure_points": ["Security breaches", "Compliance violations", "System downtime", "Budget overruns"],
                "communication_style": "Formal, data-driven, detail-oriented",
                "priorities": [
                    "Security and compliance first",
                    "Minimizing business disruption", 
                    "Cost predictability",
                    "Vendor reliability and support"
                ],
                "common_objections": [
                    "How do we know this is secure?",
                    "What if something goes wrong?",
                    "Do you have references from similar companies?",
                    "What's the total cost over 3 years?"
                ]
            },
            
            "innovation_cio": {
                "name": "Marcus Rodriguez", 
                "role": "Chief Information Officer",
                "company_context": {
                    "size": "mid-market",
                    "industry": "technology",
                    "challenges": ["rapid scaling", "competitive differentiation", "talent acquisition"]
                },
                "decision_style": "Quick decision-maker, embraces calculated risks, innovation-focused",
                "personality_bias": "Forward-thinking, competitive, values cutting-edge solutions",
                "pressure_points": ["Falling behind competitors", "Slow time-to-market", "Scaling challenges"],
                "communication_style": "Direct, strategic, focuses on outcomes",
                "priorities": [
                    "Competitive advantage through technology",
                    "Speed and agility",
                    "Scalability and future-proofing",
                    "Team empowerment and productivity"
                ],
                "common_objections": [
                    "How does this make us faster than competitors?",
                    "Will this scale as we grow?",
                    "What's the implementation timeline?",
                    "How do we measure ROI?"
                ]
            },
            
            "budget_conscious_cio": {
                "name": "Jennifer Walsh",
                "role": "Chief Information Officer", 
                "company_context": {
                    "size": "small-medium business",
                    "industry": "manufacturing",
                    "challenges": ["cost optimization", "operational efficiency", "limited IT resources"]
                },
                "decision_style": "Cost-benefit focused, practical, values simplicity",
                "personality_bias": "Pragmatic, budget-conscious, prefers simple solutions",
                "pressure_points": ["Cost overruns", "Resource constraints", "Complexity management"],
                "communication_style": "Straightforward, practical, ROI-focused", 
                "priorities": [
                    "Clear ROI and cost justification",
                    "Ease of implementation and use",
                    "Minimal ongoing maintenance",
                    "Reliable vendor support"
                ],
                "common_objections": [
                    "What's the real total cost?",
                    "How much time will implementation take?",
                    "Do we really need all these features?",
                    "What if we start small?"
                ]
            }
        }
        
        # CFO Personas  
        cfo_personas = {
            "roi_focused_cfo": {
                "name": "David Kim",
                "role": "Chief Financial Officer",
                "company_context": {
                    "size": "large enterprise", 
                    "industry": "retail",
                    "challenges": ["margin pressure", "digital transformation costs", "economic uncertainty"]
                },
                "decision_style": "Numbers-driven, requires detailed financial analysis",
                "personality_bias": "Analytical, skeptical of unproven ROI claims",
                "pressure_points": ["Quarterly earnings pressure", "Cost management", "Investment justification"],
                "communication_style": "Data-heavy, precise, focuses on financial metrics",
                "priorities": [
                    "Measurable financial returns",
                    "Cost optimization opportunities", 
                    "Risk mitigation",
                    "Operational efficiency gains"
                ],
                "common_objections": [
                    "Show me the detailed ROI calculation",
                    "What are the hidden costs?",
                    "How do we measure success?",
                    "What's the payback period?"
                ]
            },
            
            "growth_cfo": {
                "name": "Lisa Thompson",
                "role": "Chief Financial Officer",
                "company_context": {
                    "size": "high-growth startup",
                    "industry": "SaaS",
                    "challenges": ["scaling operations", "cash flow management", "investor expectations"]
                },
                "decision_style": "Growth-oriented, comfortable with calculated risks",
                "personality_bias": "Strategic, values speed and scalability",
                "pressure_points": ["Growth targets", "Cash burn rate", "Market competition"],
                "communication_style": "Strategic, forward-looking, growth-focused",
                "priorities": [
                    "Revenue growth acceleration",
                    "Operational scalability",
                    "Competitive positioning",
                    "Investor appeal"
                ],
                "common_objections": [
                    "How does this accelerate growth?",
                    "What's the revenue impact?",
                    "How quickly can we scale this?",
                    "What do investors think of this approach?"
                ]
            }
        }
        
        # CMO Personas
        cmo_personas = {
            "brand_focused_cmo": {
                "name": "Amanda Foster",
                "role": "Chief Marketing Officer", 
                "company_context": {
                    "size": "large enterprise",
                    "industry": "consumer goods",
                    "challenges": ["brand differentiation", "customer loyalty", "market saturation"]
                },
                "decision_style": "Brand-first, values creative and strategic alignment",
                "personality_bias": "Creative, brand-protective, values consistency",
                "pressure_points": ["Brand reputation", "Customer perception", "Market share"],
                "communication_style": "Creative, brand-focused, emotionally intelligent",
                "priorities": [
                    "Brand consistency and strength",
                    "Customer experience quality",
                    "Creative differentiation", 
                    "Long-term brand value"
                ],
                "common_objections": [
                    "How does this align with our brand?",
                    "What's the customer experience impact?", 
                    "Will this differentiate us creatively?",
                    "How do we maintain brand consistency?"
                ]
            },
            
            "data_driven_cmo": {
                "name": "Ryan Mitchell",
                "role": "Chief Marketing Officer",
                "company_context": {
                    "size": "digital-first company",
                    "industry": "e-commerce", 
                    "challenges": ["attribution complexity", "customer acquisition costs", "personalization at scale"]
                },
                "decision_style": "Data-driven, performance-focused, test-and-learn approach",
                "personality_bias": "Analytical, performance-oriented, values measurability",
                "pressure_points": ["CAC vs LTV", "Attribution accuracy", "Campaign performance"],
                "communication_style": "Metrics-focused, performance-oriented, testing-minded",
                "priorities": [
                    "Measurable performance improvement",
                    "Customer acquisition efficiency",
                    "Data accuracy and insights",
                    "Scalable optimization"
                ],
                "common_objections": [
                    "How do we measure the impact?",
                    "What's the effect on CAC?",
                    "Can we A/B test this?", 
                    "How does this improve conversion?"
                ]
            }
        }
        
        # Combine all personas
        all_personas = {}
        all_personas.update(cio_personas)
        all_personas.update(cfo_personas) 
        all_personas.update(cmo_personas)
        
        return {key: AIPersona(data) for key, data in all_personas.items()}
    
    def get_personas_by_role(self, role_filter=None):
        """Get personas filtered by role"""
        if role_filter:
            return {k: v for k, v in self.personas.items() if role_filter.lower() in v.role.lower()}
        return self.personas
    
    def test_message_with_persona(self, message, persona, openai_client, evaluation_criteria):
        """Test a message with a specific persona using AI"""
        if not openai_client:
            return self._fallback_evaluation(message, persona, evaluation_criteria)
            
        try:
            # Create the evaluation prompt
            system_prompt = persona.get_system_prompt()
            
            user_prompt = f"""Please evaluate this marketing message:

"{message}"

Rate each criterion from 1-10 and provide specific feedback:

EVALUATION CRITERIA:
{chr(10).join([f"- {criterion}: Rate 1-10" for criterion in evaluation_criteria])}

Please respond in this EXACT JSON format:
{{
    "overall_impression": "Your initial reaction as {persona.name}",
    "scores": {{
        {chr(10).join([f'        "{criterion.lower().replace(" ", "_")}": 0' for criterion in evaluation_criteria])}
    }},
    "detailed_feedback": {{
        {chr(10).join([f'        "{criterion.lower().replace(" ", "_")}": "Specific feedback on {criterion.lower()}"' for criterion in evaluation_criteria])}
    }},
    "key_concerns": ["concern1", "concern2", "concern3"],
    "recommendations": ["recommendation1", "recommendation2"],
    "likelihood_to_act": 0,
    "decision_factors": ["What would make this more compelling for you"],
    "persona_authentic_response": "How you would actually respond to this message in a meeting"
}}"""

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                result = json.loads(json_text)
                result['persona_name'] = persona.name
                result['persona_role'] = persona.role
                result['success'] = True
                return result
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a fallback response
                return {
                    'persona_name': persona.name,
                    'persona_role': persona.role, 
                    'overall_impression': response_text[:200] + "...",
                    'scores': {criterion.lower().replace(" ", "_"): 5 for criterion in evaluation_criteria},
                    'success': False,
                    'error': 'Could not parse AI response'
                }
                
        except Exception as e:
            return self._fallback_evaluation(message, persona, evaluation_criteria, str(e))
    
    def _fallback_evaluation(self, message, persona, evaluation_criteria, error=None):
        """Provide a fallback evaluation when AI is not available"""
        return {
            'persona_name': persona.name,
            'persona_role': persona.role,
            'overall_impression': f"As {persona.name}, I would need to review this message more carefully. Consider our priorities: {', '.join(persona.priorities[:2])}.",
            'scores': {criterion.lower().replace(" ", "_"): 6 for criterion in evaluation_criteria},
            'detailed_feedback': {
                criterion.lower().replace(" ", "_"): f"AI analysis not available. Consider how this addresses {persona.priorities[0].lower()}."
                for criterion in evaluation_criteria
            },
            'key_concerns': persona.common_objections[:3],
            'recommendations': ["Test with AI analysis enabled", "Consider persona priorities"],
            'likelihood_to_act': 5,
            'decision_factors': persona.priorities[:2],
            'persona_authentic_response': f"I'd need more information about how this addresses {persona.pressure_points[0].lower()}.",
            'success': False,
            'error': error or 'AI analysis not available'
        }

def render_persona_testing_interface():
    """Render the main persona testing interface"""
    
    st.markdown("## 🎭 AI Persona Message Testing")
    st.markdown("Test your messaging against AI decision-maker personas representing your target audience.")
    
    # Initialize the testing system
    testing_system = PersonaTestingSystem()
    
    # Message input section
    st.subheader("📝 Message to Test")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        message_input = st.text_area(
            "Enter your marketing message:",
            placeholder="Enter the message you want to test with AI decision-makers...",
            height=120,
            key="persona_message_input"
        )
    
    with col2:
        st.markdown("**Message Examples:**")
        
        if st.button("💼 B2B SaaS", key="example_b2b"):
            st.session_state.persona_message_input = "Transform your business operations with our AI-powered platform. Reduce costs by 30% while increasing productivity. Trusted by Fortune 500 companies worldwide."
            st.rerun()
            
        if st.button("🔒 Cybersecurity", key="example_security"):
            st.session_state.persona_message_input = "Protect your organization with enterprise-grade security that adapts to emerging threats. Zero-trust architecture with 99.9% uptime guarantee."
            st.rerun()
            
        if st.button("📊 Analytics", key="example_analytics"):
            st.session_state.persona_message_input = "Make data-driven decisions with real-time insights. Our platform integrates all your data sources and delivers actionable intelligence to drive growth."
            st.rerun()
    
    # Persona selection
    st.subheader("🎯 Select Decision-Maker Personas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        role_filter = st.selectbox(
            "Filter by Role:",
            ["All Roles", "Chief Information Officer", "Chief Financial Officer", "Chief Marketing Officer"],
            key="role_filter"
        )
    
    with col2:
        evaluation_criteria = st.multiselect(
            "Evaluation Criteria:",
            ["Clarity", "Relevance", "Credibility", "Differentiation", "Urgency", "Trust", "Value Proposition"],
            default=["Clarity", "Relevance", "Credibility", "Differentiation"],
            key="eval_criteria"
        )
    
    # Get filtered personas
    filter_role = None if role_filter == "All Roles" else role_filter
    available_personas = testing_system.get_personas_by_role(filter_role)
    
    # Persona selection
    selected_persona_keys = st.multiselect(
        "Choose Personas to Test:",
        options=list(available_personas.keys()),
        format_func=lambda x: f"{available_personas[x].name} - {available_personas[x].role}",
        default=list(available_personas.keys())[:3],
        key="selected_personas"
    )
    
    # AI Configuration
    st.subheader("🤖 AI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_ai_analysis = st.checkbox(
            "🧠 Enable AI-Powered Persona Analysis",
            value=True,
            help="Use AI to generate authentic persona responses"
        )
    
    with col2:
        if use_ai_analysis:
            openai_api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Required for AI persona analysis"
            )
        else:
            openai_api_key = None
    
    # Testing button
    if st.button("🧪 Test Message with Personas", type="primary", disabled=not message_input or not selected_persona_keys):
        
        if not evaluation_criteria:
            st.error("Please select at least one evaluation criterion.")
            return
            
        # Initialize OpenAI client if available
        openai_client = None
        if use_ai_analysis and openai_api_key and OPENAI_AVAILABLE:
            try:
                openai_client = openai.OpenAI(api_key=openai_api_key)
            except Exception as e:
                st.warning(f"Could not initialize OpenAI client: {e}")
        
        # Run the tests
        with st.spinner("Testing message with AI personas..."):
            results = []
            
            progress_bar = st.progress(0)
            
            for i, persona_key in enumerate(selected_persona_keys):
                persona = available_personas[persona_key]
                
                result = testing_system.test_message_with_persona(
                    message_input, 
                    persona, 
                    openai_client, 
                    evaluation_criteria
                )
                
                results.append(result)
                progress_bar.progress((i + 1) / len(selected_persona_keys))
            
            progress_bar.empty()
        
        # Store results in session state
        st.session_state.persona_test_results = {
            'message': message_input,
            'criteria': evaluation_criteria,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Display results
        render_persona_test_results(results, evaluation_criteria, message_input)
    
    # Display previous results if available
    if 'persona_test_results' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Previous Test Results")
        
        if st.button("🔄 Show Previous Results"):
            prev_results = st.session_state.persona_test_results
            render_persona_test_results(
                prev_results['results'], 
                prev_results['criteria'], 
                prev_results['message']
            )

def render_persona_test_results(results, criteria, message):
    """Render the persona testing results"""
    
    st.markdown("---")
    st.subheader("🎯 Persona Testing Results")
    
    # Overview metrics
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        avg_scores = {}
        for criterion in criteria:
            criterion_key = criterion.lower().replace(" ", "_")
            scores = [r['scores'].get(criterion_key, 0) for r in successful_results]
            avg_scores[criterion] = np.mean(scores) if scores else 0
        
        overall_avg = np.mean(list(avg_scores.values())) if avg_scores else 0
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{overall_avg:.1f}/10")
        
        with col2:
            st.metric("Personas Tested", len(results))
        
        with col3:
            high_scores = sum(1 for r in successful_results if np.mean(list(r['scores'].values())) >= 7)
            st.metric("High Ratings (7+)", f"{high_scores}/{len(successful_results)}")
        
        with col4:
            likelihood_scores = [r.get('likelihood_to_act', 0) for r in successful_results]
            avg_likelihood = np.mean(likelihood_scores) if likelihood_scores else 0
            st.metric("Avg Likelihood to Act", f"{avg_likelihood:.1f}/10")
    
    # Persona Resonance Map
    if successful_results:
        st.subheader("🗺️ Persona Resonance Map")
        
        # Create heatmap data
        persona_names = [r['persona_name'] for r in successful_results]
        criteria_data = []
        
        for criterion in criteria:
            criterion_key = criterion.lower().replace(" ", "_")
            scores = [r['scores'].get(criterion_key, 0) for r in successful_results]
            criteria_data.append(scores)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=criteria_data,
            x=persona_names,
            y=criteria,
            colorscale='RdYlGn',
            colorbar=dict(title="Score (1-10)"),
            text=[[f"{score:.1f}" for score in row] for row in criteria_data],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Persona Response Heatmap",
            xaxis_title="AI Personas",
            yaxis_title="Evaluation Criteria",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual Persona Results
    st.subheader("👤 Individual Persona Feedback")
    
    for i, result in enumerate(results):
        with st.expander(f"🎭 {result['persona_name']} - {result['persona_role']}", expanded=i<2):
            
            if result.get('success', False):
                # Overall impression
                st.markdown(f"**Initial Reaction:** {result['overall_impression']}")
                
                # Scores
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Scores:**")
                    for criterion in criteria:
                        criterion_key = criterion.lower().replace(" ", "_")
                        score = result['scores'].get(criterion_key, 0)
                        color = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                        st.write(f"{color} {criterion}: {score}/10")
                
                with col2:
                    likelihood = result.get('likelihood_to_act', 0)
                    likelihood_color = "🟢" if likelihood >= 7 else "🟡" if likelihood >= 5 else "🔴"
                    st.markdown(f"**🎯 Likelihood to Act:** {likelihood_color} {likelihood}/10")
                
                # Detailed feedback
                if 'detailed_feedback' in result:
                    st.markdown("**💬 Detailed Feedback:**")
                    for criterion in criteria:
                        criterion_key = criterion.lower().replace(" ", "_")
                        feedback = result['detailed_feedback'].get(criterion_key, "No feedback available")
                        st.write(f"• **{criterion}:** {feedback}")
                
                # Key concerns and recommendations
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'key_concerns' in result and result['key_concerns']:
                        st.markdown("**⚠️ Key Concerns:**")
                        for concern in result['key_concerns']:
                            st.write(f"• {concern}")
                
                with col2:
                    if 'recommendations' in result and result['recommendations']:
                        st.markdown("**💡 Recommendations:**")
                        for rec in result['recommendations']:
                            st.write(f"• {rec}")
                
                # Authentic response
                if 'persona_authentic_response' in result:
                    st.markdown("**🎙️ Authentic Response:**")
                    st.info(f'"{result["persona_authentic_response"]}"')
                    
            else:
                st.warning(f"AI analysis failed: {result.get('error', 'Unknown error')}")
                if 'overall_impression' in result:
                    st.write(result['overall_impression'])
    
    # Action Items and Summary
    if successful_results:
        st.subheader("📋 Action Items & Summary")
        
        # Aggregate insights
        all_concerns = []
        all_recommendations = []
        
        for result in successful_results:
            all_concerns.extend(result.get('key_concerns', []))
            all_recommendations.extend(result.get('recommendations', []))
        
        # Most common concerns
        if all_concerns:
            concern_counts = pd.Series(all_concerns).value_counts()
            st.markdown("**🔍 Most Common Concerns:**")
            for concern, count in concern_counts.head(5).items():
                st.write(f"• {concern} (mentioned by {count} personas)")
        
        # Most common recommendations  
        if all_recommendations:
            rec_counts = pd.Series(all_recommendations).value_counts()
            st.markdown("**✨ Top Recommendations:**")
            for rec, count in rec_counts.head(5).items():
                st.write(f"• {rec} (suggested by {count} personas)")
    
    # Export functionality
    st.subheader("📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download as CSV"):
            # Create DataFrame for export
            export_data = []
            for result in results:
                row = {
                    'persona_name': result['persona_name'],
                    'persona_role': result['persona_role'],
                    'overall_impression': result.get('overall_impression', ''),
                    'likelihood_to_act': result.get('likelihood_to_act', 0)
                }
                
                # Add scores
                if 'scores' in result:
                    for criterion in criteria:
                        criterion_key = criterion.lower().replace(" ", "_")
                        row[f'score_{criterion}'] = result['scores'].get(criterion_key, 0)
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "persona_test_results.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("📋 Copy Summary"):
            summary = f"""
PERSONA TESTING SUMMARY
Message: "{message}"
Tested: {datetime.now().strftime('%Y-%m-%d %H:%M')}

OVERALL RESULTS:
- Personas Tested: {len(results)}
- Average Score: {overall_avg:.1f}/10
- High Performers (7+): {high_scores}/{len(successful_results)}

TOP CONCERNS:
{chr(10).join([f"• {concern}" for concern in concern_counts.head(3).index])}

TOP RECOMMENDATIONS:
{chr(10).join([f"• {rec}" for rec in rec_counts.head(3).index])}
"""
            st.code(summary)
